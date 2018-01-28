import cv2
import sys
import os
import imageio
import glob
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib

import kcftracker
from sift import SIFT
from MLP import MLP_Detection, MLP_Detection_MP
from video import Video
from config import *

# Version check
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def cropImageAndHistogram(image, bbox, HOG=False):
    # Crop patch and analysis using histogram
    h, w = image.shape[:2]

    crop_x_min = int(max(0, bbox[0] - PATCH_MARGIN))
    crop_x_max = int(min(w - 1, bbox[0] + bbox[2] + PATCH_MARGIN))
    crop_y_min = int(max(0, bbox[1] - PATCH_MARGIN))
    crop_y_max = int(min(h - 1, bbox[1] + bbox[3] + PATCH_MARGIN))

    patch = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    if patch.shape[0] != bbox[3] or patch.shape[1] != bbox[2]: # image edge case
        return None

    if not HOG:
        hist = np.array(patch).astype(float) / 255
        hist -= np.mean(hist)
    else:
        hist = [hog(patch[..., i], orientations=9, 
                                   pixels_per_cell=(8, 8), 
                                   cells_per_block=(3, 3), 
                                   block_norm="L2", 
                                   visualise=False) for i in range(3)]
    hist = np.array(hist)
    hist = hist.reshape(hist.size)
    return hist

def computeHistDist(currHist, prevHist):
    return np.linalg.norm(np.abs(currHist - prevHist))

def cropImageAndAnalysis(clf, image, bbox, prevHist, HOG=False, USE_CLF=True):
    assert clf is not None, "No classifier loaded!"

    hist = cropImageAndHistogram(image, bbox, HOG)
    if hist is None: # crop image size is incorrect (near the edge)
        return False, prevHist, None
    if USE_CLF:
        pred = clf.predict([hist])
        dist = computeHistDist(hist, prevHist)
        if pred == 1: # find object
            return True, hist, dist # for now, don't count the score in.
            # score = clf.predict_proba(fd)[0][pred]
            # score = clf.decision_function([hist])
            # if score > SCORE_CRITERIA:
            #     return True, hist, dist
            # else:
            #     return False, hist, dist
        else:
            return False, hist, dist
    else:
        dist = computeHistDist(hist, prevHist)
        prevHist = hist
        if dist > THRESHOLD_VALUE:
            return False, hist, dist
        else: 
            return True, hist, dist

def drawBox(image, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(image, p1, p2, (255,255,255), 2, 2)
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
            # tracker = cv2.TrackerKCF_create()
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
    tmp = frame_resize[..., 0].copy()
    frame_resize[..., 0] = frame_resize[..., 2].copy()
    frame_resize[..., 2] = tmp
    return frame_resize