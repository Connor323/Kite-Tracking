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
    KILL_BS[0] = True
    # sys.exit(0)

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
    frame_show = cv2.resize(frame, VIZ_SIZE)
    TRACKING_RECORD[0] = frame_show
    frame_show = getResultFrame()

    if SHOW_RESULT:
        cv2.imshow("Tracking", frame_show)

    frame_resize = cv2.resize(frame_show, RECORD_SIZE)
    frame_resize = swapChannels(frame_resize)

    return frame_resize

def drawAnlge(frame, angle, center, length=25):
    """
    Draw angle axis.

    Params:
        frame: current image
        angle: float value
        center: centroid
        length: length of axis
    Return:
        Painted image
    """
    center = tuple(center)
    radian = angle / 180 * np.pi
    vertice = (int(center[0] + np.cos(radian)*length), int(center[1] + np.sin(radian)*length))
    cv2.line(frame, center, vertice, (0, 0, 255), 5)
    return frame

def drawPoint(frame, point, color=(255, 0, 255), radius=10):
    """
    Draw single point.
    Params:
        frame: current image
        point: 2D point in pixel
        color: B G R
        radius: radius of the circle 
    Return:
        Painted image
    """
    return cv2.circle(frame, tuple(point), radius, color, -1)

def savePatchPerAngle(frame, angle, bbox):
    """
    Save the image patch for training CNN. Save the patches based on 
    the number of class in separate directories. 
    Params:
        frame: current image
        angle: output estimated angle
        bbox: bounding box
    """
    division = 360 / NUM_DIVISION_SAMPLES
    angle_idx = int(np.round(angle / division, 0)) % NUM_DIVISION_SAMPLES
    DIR = os.path.join(RESULT_BASE, "angle_%d" % angle_idx)
    # DIR_tmp = os.path.join(RESULT_BASE, "angle")
    img = cropImage(frame, bbox)
    cv2.imwrite(os.path.join(DIR, "image_%05d.png" % SAMPLE_COUNTER[angle_idx]), img)
    # cv2.imwrite(os.path.join(DIR_tmp, "image_%05d.png" % SAMPLE_COUNTER[-1]), img)
    SAMPLE_COUNTER[angle_idx] += 1
    SAMPLE_COUNTER[-1] += 1

