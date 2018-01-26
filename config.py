import os
import numpy as np
from sklearn.externals import joblib

# Select tracker 
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[1]

# image and template path
# IMAGE_PATH = "../images/clear0/*.bmp"
IMAGE_PATH = "/Users/hanxiang/Dropbox/20180118/*.bmp"
# IMAGE_PATH = "../images/cloudy0/*.bmp"
TEMPLATE_PATH = "templates/kite0/*.png"
# TEMPLATE_PATH = "templates/kite1/*.png"
# File format
# NOTE: Format 0: 2018-1-18-12-49-0-204-original.bmp
#       Format 1: 2017-12-15-10-32-8-595.bmp (without "original")
FILE_FORMAT = 0

# The margin when crop the image for histogram computation
PATCH_MARGIN = 0 
THRESHOLD_VALUE = 3000 # TODO: dynamic select this value if possible (find new criteria)
HOG = True

USE_CLF = True
SVM_MODEL_PATH = "model/mlp.model"
clf = None
SCORE_CRITERIA = 1.0 # SVM
PROB_CRITERIA = 0.9999 # MLP
if os.path.exists(SVM_MODEL_PATH):
    clf = joblib.load(SVM_MODEL_PATH)

# Multi-thread boost setting 
NUM_THREADS = 8
DO_MP = True

# Define an initial bounding box
ROI = [489, 1230, 1407, 609] # The search area when we fail on tracking.
init_bbox = None # None, if no initial bbox
DEFAULT_BBOX = [0, 0, 50, 50] # If init_bbox is none, we use the size of defalt bbox for following tracking
# bbox = (610,  1315, 51, 37) # clear0
# bbox = (1194, 1686, 21, 34) # cloudy0

# Record setting
RECORD_SIZE = (512, 512)