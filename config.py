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
# IMAGE_PATH = "../images/cloudy0/*.bmp"
TEMPLATE_PATH = "templates/kite0/*.png"

START_FRAME = None # the path to the start frame name, in case we want to start in the middle of video
				   # Set None if we want to stat from beginning. 

# File format
# NOTE: Format 0: 2018-1-18-12-49-0-204-original.bmp
#       Format 1: 2017-12-15-10-32-8-595.bmp (without "original")
FILE_FORMAT = 0

# Background Substraction setting 
fgbg_kernel_close_size = 3 # for morphological closing and opening 
fgbg_kernel_open_size = 3 # for morphological closing and opening 
history_length = 100 # buffer of history
fgbg_kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fgbg_kernel_close_size, fgbg_kernel_close_size))
fgbg_kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fgbg_kernel_open_size, fgbg_kernel_open_size))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=history_length)

MIN_AREA = 25 # minimum area inside bbox for BS
MAX_AREA = 600 # maximum area inside bbox for BS

TRACKING_CRITERIA_AREA = 100 # minimum area inside bbox for tracking
BBOX_TWIST_SIZE = 5
    
# Classifier setting 
MLP_MODEL_PATH = "model/mlp_1layer.model"
BG_MODEL_PATH  = "model/mlp_bg.model" 

clf = joblib.load(MLP_MODEL_PATH) # MLP_1 for initial bbox detection 
bg_clf = joblib.load(BG_MODEL_PATH) # MLP_2 for BS detection

PROB_CRITERIA = 0.95 # The prob_thresh value for MLP_2

# Multi-thread boost setting 
NUM_THREADS = 16

# Define an initial bounding box
ROI = [489, 1230, 1407, 609] # The search area when we fail on tracking.
init_bbox = None # Use None, if no initial bbox
BBOX_SIZE = [51, 51] # If init_bbox is none, we use the size of defalt bbox for following tracking
STEP_SIZE = [51, 51]

# Record setting
RECORD_SIZE = (512, 512)
RECORD_FPS = 100

# Debug setting
DEBUG_MODE = True