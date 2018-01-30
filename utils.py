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

    fgmask = fgbg.apply(image_resize)
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
        tmp_show = cv2.resize(tmp, RECORD_SIZE, cv2.INTER_NEAREST)
        cv2.imshow("BS", tmp_show)
    
    if not return_centroids:
        return final_labels
    else:
        centroids = centroids[select_labels]
        centroids *= 2
        return final_labels, centroids.astype(int)

def twistAndCrop(image, bbox):
    """
    Twist the bbox to find the one with maximum area of foreground.

    Params:
        image: BS result (binary image)
        bbox: bounding box
    Return:
        refined bbox
        patch
    """
    h, w = image.shape[:2]
    twists = [[0, 0], [1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [-1, -1], [1, -1], [-1, 1]]
    
    bbox = np.array(bbox)
    patch = None
    refined_bbox = None
    max_area = -1

    for twist in twists:
        tmp_bbox = bbox
        tmp_bbox[2:] = bbox[2:] + np.array(twist) * BBOX_TWIST_SIZE

        crop_x_min = int(max(0, tmp_bbox[0]))
        crop_x_max = int(min(w - 1, tmp_bbox[0] + tmp_bbox[2]))
        crop_y_min = int(max(0, tmp_bbox[1]))
        crop_y_max = int(min(h - 1, tmp_bbox[1] + tmp_bbox[3]))
        patch = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        if patch.shape[0] != bbox[3] or patch.shape[1] != bbox[2]: # image edge case
            patch = None
            continue
        else:
            area = np.sum(patch != 0)
            if area > max_area:
                max_area = area
                refined_bbox = tmp_bbox
    return patch, refined_bbox

def cropImageFromBS(image, bbox):
    """
    Crop patch and analysis using histogram

    Params: 
        image: current frame
        bbox: bounding box
    """
    image = process_bs(image, low_area=MIN_AREA, up_area=MAX_AREA)
    patch, refined_bbox = twistAndCrop(image, bbox)

    return patch, refined_bbox

def cropImageAndAnalysis(clf, image, bbox):
    """
    Determine if the current patch contains target

    Params: 
        image: current frame
        bbox: bounding box
    """
    assert clf is not None, "No classifier loaded!"

    patch, refined_bbox = cropImageFromBS(image, bbox)
    if patch is None: # crop image size is incorrect (near the edge)
        return False, None
    if np.sum(patch != 0) > TRACKING_CRITERIA_AREA:
        return True, refined_bbox
    return False, None

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

def displayFrame(frame, resize=(250, 250)):
    frame_resize = cv2.resize(frame, resize)
    cv2.imshow("Tracking", frame_resize)
    frame_resize = swapChannels(frame_resize)
    return frame_resize