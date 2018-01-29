import os
import pickle
import numpy as np
import cv2
from sklearn.externals import joblib

# Select tracker 
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]

# image and template path
IMAGE_PATH = "/Users/hanxiang/Dropbox/20180118/*.bmp"
TEMPLATE_PATH = "templates/kite0/*.png"

START_FRAME = None # the path to the start frame name, in case we want to start in the middle of video

# File format
# NOTE: Format 0: 2018-1-18-12-49-0-204-original.bmp
#       Format 1: 2017-12-15-10-32-8-595.bmp (without "original")
FILE_FORMAT = 0

# Background Substraction setting 
fgbg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

MIN_AREA = 50
MAX_AREA = 1000
    
# Classifier setting 
MLP_MODEL_PATH = "model/mlp_1layer.model"
BG_MODEL_PATH  = "model/mlp_bg.model" 

clf = joblib.load(MLP_MODEL_PATH) # MLP_1 for initial bbox detection 
bg_clf = joblib.load(BG_MODEL_PATH) # MLP_2 for BS detection

PROB_CRITERIA = 0.98 # The prob_thresh value for MLP_2

# Multi-thread boost setting 
NUM_THREADS = 8

# Define an initial bounding box
ROI = [489, 1230, 1407, 609] # The search area when we fail on tracking.
init_bbox = None # Use None, if no initial bbox
BBOX_SIZE = [51, 51] # If init_bbox is none, we use the size of defalt bbox for following tracking
STEP_SIZE = [51, 51]

# Record setting
RECORD_SIZE = (512, 512)

# Debug setting
DEBUG_MODE = True