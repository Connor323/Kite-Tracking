import cv2
import sys
import os
import imageio
import glob
import numpy as np
import signal
import time
from skimage.feature import hog

import kcftracker
from video import Video
from config import *

# Version check
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def signal_handler(signal, frame):
    """
    handle the case when hit Ctrl+C
    """
    print('   -> Stop the program by Ctrl+C! ')
    if not DEBUG_MODE:
        print("Bounding box saves to {}".format(TARGET_BOX))
        BBOX_FILE.close()
    sys.exit(0)

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

def process_bs(image, low_area=10, up_area=1000, return_centroids=False):
    """
    This function applies the BS model given the current frame and implements the morphological 
    operation and region growing as the post process. 

    Params: 
        image: current frame 
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
    image_resize = cv2.resize(image, (int(w // BS_DOWNSAMPLE), int(h // BS_DOWNSAMPLE)))
    ROI_Y = int(len(image_resize) * HEIGHT_ROI_RATIO)
    image_resize = image_resize[ROI_Y:]

    # Apply BS 
    if BG_MODEL[0] is not None:
        fgmask = fgbg.apply(image_resize, BG_MODEL[0])
    else:
        fgmask = fgbg.apply(image_resize)

    if DEBUG_MODE:
        tmp_show = cv2.resize(fgmask, VIZ_SIZE, cv2.INTER_NEAREST)
        BS_ORIGIN_RECORD[0] = cv2.cvtColor(tmp_show, cv2.COLOR_GRAY2RGB)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, fgbg_kernel_open) # remove small items 
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, fgbg_kernel_close) # fill holes
    
    # obtain the regions in range of area (low_area, up_area)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask.astype(np.uint8), connectivity=4)
    select_labels = np.where((stats[..., -1] > low_area) * \
                             (stats[..., -1] < up_area))[0]

    # refine the labels
    new_h, new_w = labels.shape[:2]
    tmp_labels = np.zeros([new_h+ROI_Y, new_w], np.uint8)
    tmp = np.zeros_like(labels).astype(np.uint8)
    for select_label in select_labels: 
        tmp[labels == select_label] = 255
    tmp_labels[ROI_Y:] = tmp
    final_labels = cv2.resize(tmp_labels, (w, h), cv2.INTER_NEAREST)
    FG_MODEL[0] = tmp

    if DEBUG_MODE:
        tmp_show = cv2.resize(tmp_labels, VIZ_SIZE, cv2.INTER_NEAREST)
        BS_POST_RECORD[0] = cv2.cvtColor(tmp_show, cv2.COLOR_GRAY2RGB)
    
    if not return_centroids:
        return final_labels
    else:
        if len(centroids):
            centroids = centroids[select_labels]
            centroids[:, 1] += ROI_Y
            centroids *= BS_DOWNSAMPLE
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
    c_bbox = np.array(nd_bbox[:2] + nd_bbox[2:] // 2)
    
    dists = np.linalg.norm(centroids - c_bbox, axis=1)
    min_idx = np.argmin(dists)
    min_val = dists[min_idx]

    if min_val > RECENTER_THRESH:
        new_bbox = nd_bbox
        new_bbox[:2] = centroids[min_idx] - nd_bbox[2:] // 2
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
    if image is None:
        return None

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
    Return:
        patch of image
        if succeed
    """
    image, centroids = process_bs(image, low_area=MIN_AREA, up_area=MAX_AREA, return_centroids=True)
    if len(centroids) > 0:
        patch, ret = centerBoxAndCrop(image, centroids, bbox)
    else:
        return None, True

    return patch, ret

def updateBG():
    """
    Update background model for BS

    Params: 
        None
    return:
        None
    """
    if BG_MODEL[0] is None: 
        BG_MODEL[0] = fgbg.getBackgroundImage()
    else:     
        if FG_MODEL[0] is not None:
            newBG = fgbg.getBackgroundImage()
            BG_MODEL[0][FG_MODEL[0] == 0] = newBG[FG_MODEL[0] == 0]

def cropImageAndAnalysis(image, bbox):
    """
    Determine if the current patch contains target

    Params: 
        image: current frame
        bbox: bounding box
    return:
        if the current tracking is successful (boolean)
        image patch from BS result
    """
    if UPDATE_BACKGROUND: 
        updateBG()

    patch, ret = cropImageFromBS(image, bbox)
    if patch is None or ret: # crop image size is incorrect (near the edge)
        return False, patch
    if np.sum(patch != 0) > TRACKING_CRITERIA_AREA:
        return True, patch
    return False, patch

def drawBox(image, bbox):
    """
    Draw bounding box.

    Params:
        bbox: [x_top_left, y_top_left, width, height]
    Return:
        image
    """
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(image, p1, p2, (0, 255, 0), 2, 2)
    return image

def creat_tracker(tracker_type):
    """
    Create video tracker.
    
    Params:
        tracker_type: string of tracker name
    Return:
        tracker object
    """
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            # tracker = cv2.TrackerKCF_create() # OpenCV KCF is not good
            tracker = kcftracker.KCFTracker(False, True, False)  # hog, fixed_window, multiscale
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    return tracker

def getResultFrame():
    """
    Stiching all the results into one image, including:
        - tracking result;
        - localization result;
        - matched filter result;
        - BS-original result;
        - BS-post result.

    Return:
        Stiching image
    """
    # Resize images
    frame_show = np.ones(list(VIZ_SIZE)+[3], np.uint8) * 255

    if type(TRACKING_RECORD[0]) == type(None) or \
       type(MLP_RECORD[0]) == type(None) or \
       type(BS_ORIGIN_RECORD[0]) == type(None) or \
       type(BS_POST_RECORD[0]) == type(None) or \
       type(KERNEL_RECORD[0]) == type(None) or \
       type(PATCH_RECORD[0]) == type(None):
        return frame_show

    tracking = cv2.resize(TRACKING_RECORD[0], (600, 600))
    localization = cv2.resize(MLP_RECORD[0], (300, 300))
    bs_original = cv2.resize(BS_ORIGIN_RECORD[0], (300, 300))
    bs_post = cv2.resize(BS_POST_RECORD[0], (300, 300))
    h1, w1 = KERNEL_RECORD[0].shape[:2]
    matched_filter = cv2.resize(KERNEL_RECORD[0], (4 * w1, 4 * h1))
    h2, w2 = PATCH_RECORD[0].shape[:2]
    matched_filter_patch = cv2.resize(PATCH_RECORD[0], (4 * w2, 4 * h2))

    # Stiching images
    frame_show[150-2*h1:150+2*h1, 150-2*w1:150+2*w1] = matched_filter
    frame_show[150-2*h2:150+2*h2, 450-2*w2:450+2*w2] = matched_filter_patch
    frame_show[-600:, :600] = tracking
    frame_show[:300, -300:] = bs_original
    frame_show[300:600, -300:] = bs_post
    frame_show[600:, -300:] = localization

    # Write labels
    cv2.putText(frame_show, "Selected Kernel", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    cv2.putText(frame_show, "Current Patch", (400,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    cv2.putText(frame_show, "BS Original", (700,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    cv2.putText(frame_show, "BS Post", (700,320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    cv2.putText(frame_show, "Localization", (700,620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)

    return frame_show

def displayFrame(frame, frame_original, bbox, angle, video):
    """
    Display/write the current tracking result or write the image patch and bbox

    Params:
        frame: current frame (with painting)
        frame_original: current frame (without painting)
        bbox: bounding box (could be None)
        angle: float (could be None)
        video: video reader object
    Return:
        Recording image
    """
    if not DEBUG_MODE:
        frame_resize = cv2.resize(frame, RECORD_SIZE)
        if bbox is None or angle is None: return frame_resize

        if WRITE_TMP_RESULT:
            cv2.imwrite("Tracking.png", frame_resize)
        else:
            cv2.imshow("Tracking", frame_resize)

        msg = "%s:%d %d %d %d %f\n" % (video.getFrameName(), bbox[0], bbox[1], bbox[2], bbox[3], angle)
        BBOX_FILE.write(msg)

        patch = cropImage(frame_original, bbox)
        if patch is not None:
            cv2.imwrite(os.path.join(TARGET_PATCH, video.getFrameName()), patch)
    else:
        frame_show = cv2.resize(frame, VIZ_SIZE)
        TRACKING_RECORD[0] = frame_show
        frame_show = getResultFrame()

        if WRITE_TMP_RESULT:
            cv2.imwrite("Tracking.png", frame_show)
        else:
            cv2.imshow("Tracking", frame_show)

        frame_resize = cv2.resize(frame_show, RECORD_SIZE)
        frame_resize = swapChannels(frame_resize)

    return frame_resize

def drawAnlge(frame, angle, bbox, length=25):
    """
    Draw angle axis.

    Params:
        frame: current image
        angle: float value
        bbox: bounding box
        length: length of axis
    Return:
        Painted image
    """
    bbox = np.array(bbox).astype(int)
    center = tuple(bbox[:2] + bbox[2:] // 2)

    radian = angle / 180 * np.pi
    vertice = (int(center[0] + np.cos(radian)*length), int(center[1] + np.sin(radian)*length))
    cv2.line(frame, center, vertice, (0, 0, 255), 5)
    return frame

def drawPoint(frame, point, color=(255, 0, 255), radius=10):
    return cv2.circle(frame, tuple(point), radius, color, -1)

