import os
import pickle
import numpy as np
import cv2
from subprocess import call
from sklearn.externals import joblib
from keras.models import load_model
import shutil
import glob

############################# Tracker Setting #############################
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]
###########################################################################

############################# Data Setting #############################
# image and template path
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# IMAGE_PATH = "/Users/hanxiang/Dropbox/20180118/*.bmp"
# IMAGE_PATH = "/Users/hanxiang/Dropbox/20180131/*.bmp"
IMAGE_PATH = "/Users/hanxiang/Dropbox/20180603/*.bmp"

TEMPLATE_PATH = os.path.join(DIR_PATH, "templates/kite0/*.png")
# KERNEL_PATH = os.path.join(DIR_PATH, "kernels/kernel_0.bmp")
# KERNEL_PATH = os.path.join(DIR_PATH, "kernels/kernel_1.bmp")
KERNEL_PATH = os.path.join(DIR_PATH, "kernels/kernel_2.bmp")

START_FRAME = None
# START_FRAME = "/Users/hanxiang/Dropbox/20180603/2018-6-3-15-24-5-366-original.bmp" # train
# START_FRAME = "/Users/hanxiang/Dropbox/20180603/2018-6-3-15-27-3-275-original.bmp" # test
# START_FRAME = "/Users/hanxiang/Dropbox/20180131/2018-1-31-10-49-22-297-original.bmp" # the path to the start frame name, in case we want to start in the middle of video
				   # Set None if we want to stat from beginning. 
# File format
# NOTE: Format 0: 2018-1-18-12-49-0-204-original.bmp
#       Format 1: 2017-12-15-10-32-8-595.bmp (without "original")
FILE_FORMAT = 0


# Classifier loading 
# MLP_MODEL_PATH = "model/mlp_1layer.model"
# BG_MODEL_PATH  = "model/mlp_bg.model" 
# BG_MODEL_PATH  = "model/mlp-bg-py3-5.model"
BG_MODEL_PATH  = "model/cnn_loc_model.h5"
ANGLE_MODEL_PATH  = "model/cnn_model-8d.h5"

# clf = joblib.load(MLP_MODEL_PATH) # MLP_1 for initial bbox detection 
ANGLE_MODEL = load_model(ANGLE_MODEL_PATH) 
bg_clf = load_model(BG_MODEL_PATH) 
# bg_clf = joblib.load(BG_MODEL_PATH) # MLP_2 for BS detection
###########################################################################

#################### Background Substraction Setting ######################
fgbg_kernel_close_size = 5 # for morphological closing and opening 
fgbg_kernel_open_size = 5 # for morphological closing and opening 
history_length = 1000 # buffer of history
fgbg_kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fgbg_kernel_close_size, fgbg_kernel_close_size))
fgbg_kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fgbg_kernel_open_size, fgbg_kernel_open_size))
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=history_length)
fgbg = cv2.createBackgroundSubtractorMOG2(history=history_length, detectShadows=False)
BG_MODEL, FG_MODEL = [None], [None]
UPDATE_BACKGROUND = False
# fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
BS_DOWNSAMPLE = 2.5
HEIGHT_ROI_RATIO = 0.3 # overall ROI starting from [HEIGHT_ROI_RATIO*h : h] in rows
INIT_FRAMES_NUM = 10 # the number of frames to skip for BS initicalization

# BS post-process setting
MIN_AREA = 30 # minimum area inside bbox for BS
MAX_AREA = 600 # maximum area inside bbox for BS

# Thread settting
KILL_BS = [False]
###########################################################################
    
############################# Tracking Setting ############################
PROB_CRITERIA = 0.50 # The prob_thresh value for MLP_2
NUM_THREADS_TRACKING = 24 # Multi-thread boost setting 

TRACKING_CRITERIA_AREA = 0 # minimum area inside bbox for tracking
RECENTER_THRESH = 15 # Max distance allowed from the centroid to the center of bbox

DECISION_BUFFER_SIZE = 3 # Decision buffer size
DECISION_BUFFER = [] # Decision buffer
BUFFER_MODE = True # If True, use the descidion buffer for tracking
###########################################################################

######################### Matched Filter Setting ##########################
NUM_ROTATION = 8 # Number of rotation for creating filter bank
THRESH_ANGLE_DISTANCE = 90 # The thresholding value for the difference of two angles in degree.
NUM_THREADS_MFR = 24 # Number of treads for computing MFR
GAIN = 0.8	# Low pass filter to remove jittering 
UPDATE_KERNEL = [False] # enable (set to True) by holding keyboard key "a" when any cv window opens 
USE_CNN = False # To enable CNN prediction, set True; otherwise, use the color-based method. 
###########################################################################

############################# BBOX Setting ################################
init_bbox = None # Use None, if no initial bbox; bbox format: [x_top_left, y_top_left, w, h]
BBOX_SIZE = [51, 51] # If init_bbox is none, we use the size of defalt bbox for following tracking
STEP_SIZE = [51, 51] # the moving step for the initial bbox detection scanning
###########################################################################

######################### Record and Debug Setting #########################
TRACKING_RECORD, KERNEL_RECORD, PATCH_RECORD, MLP_RECORD, BS_ORIGIN_RECORD, BS_POST_RECORD =\
                    								 [None], [None], [None], [None], [None], [None]
RECORD_SIZE = (912, 912) # Record image size (Don't change)
VIZ_SIZE = (900, 900) # Visulattion image size (Don't change)
RECORD_FPS = 15 # frame per second
RESULT_BASE = "result" 
CREATE_SAMPLES = False # If creating samples for training CNN
CONTINUE_CAREATE_SAMPLES = True
NUM_DIVISION_SAMPLES = 4
SAMPLE_COUNTER = []
if CREATE_SAMPLES: 
	for i in range(NUM_DIVISION_SAMPLES): 
		PATH = os.path.join(RESULT_BASE, "angle_%d" % i)
		if CONTINUE_CAREATE_SAMPLES and os.path.exists(PATH):
			fnames = glob.glob(os.path.join(PATH, "*.png"))
			if len(fnames): 
				fnames = sorted(fnames)
				fname = fnames[-1]
				last_idx = int(fname.split("_")[-1].split(".")[0])
			else:
				last_idx = -1
			SAMPLE_COUNTER.append(last_idx + 1) 
		else:
			if os.path.exists(PATH): 
				shutil.rmtree(PATH)
			SAMPLE_COUNTER.append(0)	
			os.mkdir(PATH) 	
	PATH = os.path.join(RESULT_BASE, "angle")
	call(["mkdir", "-p", PATH])
	SAMPLE_COUNTER.append(0)	

SHOW_RESULT = False # if True, will showing the image in windows
				    # if False, will disable showing
DEBUG_MODE = False # if True, will show tracking info on terminal;
				   # if False, will disable info printing

if SHOW_RESULT:
	cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
