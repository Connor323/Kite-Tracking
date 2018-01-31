import os
import pickle
import numpy as np
import cv2
from sklearn.externals import joblib

############################# Tracker Setting #############################
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]
###########################################################################

############################# Data Setting #############################
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

# Classifier loading 
MLP_MODEL_PATH = "model/mlp_1layer.model"
BG_MODEL_PATH  = "model/mlp_bg.model" 

clf = joblib.load(MLP_MODEL_PATH) # MLP_1 for initial bbox detection 
bg_clf = joblib.load(BG_MODEL_PATH) # MLP_2 for BS detection
###########################################################################

#################### Background Substraction Setting ######################
fgbg_kernel_close_size = 3 # for morphological closing and opening 
fgbg_kernel_open_size = 3 # for morphological closing and opening 
history_length = 100 # buffer of history
fgbg_kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fgbg_kernel_close_size, fgbg_kernel_close_size))
fgbg_kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fgbg_kernel_open_size, fgbg_kernel_open_size))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=history_length)

# BS post-process setting
MIN_AREA = 25 # minimum area inside bbox for BS
MAX_AREA = 600 # maximum area inside bbox for BS
###########################################################################
    
############################# Tracking Setting ############################
PROB_CRITERIA = 0.95 # The prob_thresh value for MLP_2
NUM_THREADS = 16 # Multi-thread boost setting 

TRACKING_CRITERIA_AREA = 100 # minimum area inside bbox for tracking
RECENTER_THRESH = 10 # Max distance allowed from the centroid to the center of bbox

DECISION_BUFFER_SIZE = 5 # Decision buffer size
DECISION_BUFFER = [] # Decision buffer
###########################################################################

############################# BBOX Setting ################################
ROI = [489, 1230, 1407, 609] # The search area when we fail on tracking.
init_bbox = None # Use None, if no initial bbox; bbox format: [x_top_left, y_top_left, w, h]
BBOX_SIZE = [51, 51] # If init_bbox is none, we use the size of defalt bbox for following tracking
STEP_SIZE = [51, 51] # the moving step for the initial bbox detection scanning
###########################################################################

######################### Record and Debug Setting #########################
RECORD_SIZE = (512, 512) # Record/visulattion image size
RECORD_FPS = 100 # frame per second

DEBUG_MODE = True # if True, will show the BS result and localization result
###########################################################################
