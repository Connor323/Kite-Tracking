import cv2
import sys
import os
import imageio
import glob
import numpy as np
from skimage.feature import hog

import kcftracker
from video import Video
from config import *

# Version check
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def pushBuffer(res):
    """
    Push the current detection result to the decision buffer and return the buffer result.

    Params:
        res: boolean
    Return:
        ret: boolean
    """
    def buffer_mode(res):
        """
        Return 
            BUFFER_MODE: 
                True if over half of buffer is True; else, return False
            Not BUFFER_MODE: 
                True if all of buffer is True;
                False if all of buffer is False;
                Else: return res.
        """
        if BUFFER_MODE:
            return np.sum(DECISION_BUFFER) > DECISION_BUFFER_SIZE / 2
        else:
            if (np.array(DECISION_BUFFER) == True).all():
                return True
            elif (np.array(DECISION_BUFFER) == False).all():
                return False
            else:
                return res
    
    def pushpop(res):
        del DECISION_BUFFER[0]
        DECISION_BUFFER.append(res)

    if len(DECISION_BUFFER) < DECISION_BUFFER_SIZE:
        DECISION_BUFFER.append(res)
        return res
    elif DECISION_BUFFER_SIZE == 0:
        return res
    else:
        ret = buffer_mode(res)
        pushpop(res)
        return ret

def swapChannels(image):
    """
    Convert BGR -> RGB
    """
    image = image.copy()
    tmp = image[..., 0].copy()
    image[..., 0] = image[..., 2].copy()
    image[..., 2] = tmp.copy()
    return image

def process_bs(image, downsample_rate=2, low_area=50, up_area=1000, return_centroids=False):
    """
    This function applies the BS model given the current frame and implements the morphological 
    operation and region growing as the post process. 

    Params: 
        image: current frame 
        downsample_rate: downsampling the image for faster speed. 
        low_area: minimum area of target
        up_area: maximum area of target
        return_centroids: if True, reture the centroid of selected areas
    Return: 
        final_labels: binary image obtained from BS
        centroids: centroid of selected areas on if return_centroids == True
    """
    # process background substraction
    h, w = image.shape[:2]
    
    # Downsample the image for faster speed
    image_resize = cv2.resize(image, (w / downsample_rate, h / downsample_rate))

    # Apply BS 
    fgmask = fgbg.apply(image_resize)
    
    if DEBUG_MODE:
        tmp_show = cv2.resize(fgmask, VIZ_SIZE, cv2.INTER_NEAREST)
        # cv2.imshow("BS Original", tmp_show)

    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, fgbg_kernel_open) # remove small items 
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, fgbg_kernel_close) # fill holes
    
    # obtain the regions in range of area (low_area, up_area)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask.astype(np.uint8), connectivity=8)
    select_labels = np.where((stats[..., -1] > low_area) * (stats[..., -1] < up_area) * centroids[..., -1] > h / downsample_rate / 2)[0]

    # refine the labels
    tmp = np.zeros_like(labels).astype(np.uint8)
    for select_label in select_labels: 
        tmp[labels == select_label] = 255
    final_labels = cv2.resize(tmp, (w, h), cv2.INTER_NEAREST)

    if DEBUG_MODE:
        tmp_show = cv2.resize(tmp, VIZ_SIZE, cv2.INTER_NEAREST)
        # cv2.imshow("BS Post", tmp_show)
    
    if not return_centroids:
        return final_labels
    else:
        centroids = centroids[select_labels]
        centroids *= 2
        return final_labels, centroids.astype(int)

def centerBoxAndCrop(image, centroids, bbox):
    """
    Center bbox to the centroid of target, if the difference is over the threshold value.

    Params:
        image: BS result (binary image)
        centroids: a list of points
        bbox: bounding box
    Return:
        patch
        if over the threshold value
    """
    h, w = image.shape[:2]
    nd_bbox = np.array(bbox)
    c_bbox = np.array(nd_bbox[:2] + nd_bbox[2:] / 2)
    
    dists = np.linalg.norm(centroids - c_bbox, axis=1)
    min_idx = np.argmin(dists)
    min_val = dists[min_idx]

    if min_val > RECENTER_THRESH:
        new_bbox = nd_bbox
        new_bbox[:2] = centroids[min_idx] - nd_bbox[2:] / 2
        new_bbox = new_bbox.tolist()
        return cropImage(image, new_bbox), True
    else:
        return cropImage(image, bbox), False

def cropImage(image, bbox):
    """
    Crop image.

    Params:
        image: BS result (binary image)
        bbox: bounding box
    Return:
        patch
    """
    h, w = image.shape[:2]

    crop_x_min = int(max(0, bbox[0]))
    crop_x_max = int(min(w - 1, bbox[0] + bbox[2]))
    crop_y_min = int(max(0, bbox[1]))
    crop_y_max = int(min(h - 1, bbox[1] + bbox[3]))
    patch = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    if patch.shape[0] != bbox[3] or patch.shape[1] != bbox[2]: # image edge case
        return None
    else:
        return patch

def cropImageFromBS(image, bbox):
    """
    Crop patch and analysis using histogram

    Params: 
        image: current frame
        bbox: bounding box
    """
    image, centroids = process_bs(image, low_area=MIN_AREA, up_area=MAX_AREA, return_centroids=True)
    if len(centroids) > 0:
        patch, ret = centerBoxAndCrop(image, centroids, bbox)
    else:
        return None, True

    return patch, ret

def cropImageAndAnalysis(clf, image, bbox):
    """
    Determine if the current patch contains target

    Params: 
        image: current frame
        bbox: bounding box
    """
    assert clf is not None, "No classifier loaded!"

    patch, ret = cropImageFromBS(image, bbox)
    if patch is None or ret: # crop image size is incorrect (near the edge)
        return False
    if np.sum(patch != 0) > TRACKING_CRITERIA_AREA:
        return True
    return False

def drawBox(image, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(image, p1, p2, (0, 255, 0), 2, 2)
    return image

def creat_tracker(tracker_type):
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            # tracker = cv2.TrackerKCF_create() # OpenCV KCF is not good
            tracker = kcftracker.KCFTracker(False, True, True)  # hog, fixed_window, multiscale
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    return tracker

def displayFrame(frame):
    frame_resize = cv2.resize(frame, VIZ_SIZE)
    # cv2.imshow("Tracking", frame_resize)
    frame_resize = cv2.resize(frame, RECORD_SIZE)
    frame_resize = swapChannels(frame_resize)
    return frame_resize