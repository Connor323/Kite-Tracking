import cv2
import sys
import os
import imageio
import glob
import numpy as np
from skimage.feature import hog
from subprocess import call

from sift import SIFT
from video import Video
from utils import *
from config import *

# Version check
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

############################# Setting ##############################
# Select tracker 
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[1]

# image and template path
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# IMAGE_PATH = "../images/clear0/*.bmp"
IMAGE_PATH = "/Users/hanxiang/Dropbox/20180118/*.bmp"
# IMAGE_PATH = "../images/cloudy0/*.bmp"
TEMPLATE_PATH = os.path.join(DIR_PATH, "templates/kite0/*.png")
# TEMPLATE_PATH = "templates/kite1/*.png"

# File format
# NOTE: Format 0: 2018-1-18-12-49-0-204-original.bmp
#       Format 1: 2017-12-15-10-32-8-595.bmp (without "original")
FILE_FORMAT = 0

# The margin when crop the image for histogram computation
PATCH_MARGIN = 10
THRESHOLD_VALUE = 3000 # TODO: dynamic select this value if possible (find new criteria)
HOG = False

# Define an initial bounding box
ROI = [489, 1230, 1407, 609] # The search area when we fail on tracking.
init_bbox = None # None, if no initial bbox
DEFAULT_BBOX = [0, 0, 50, 50] # If init_bbox is none, we use the size of defalt bbox for following tracking
# bbox = (610,  1315, 51, 37) # clear0
# bbox = (1194, 1686, 21, 34) # cloudy0

# Sample setting
REMOVE_OLD = True
POS_SAMPLE_PATH = "samples/pos"
NEG_SAMPLE_PATH = "samples/neg"
if REMOVE_OLD:
    call(["rm", "-r", POS_SAMPLE_PATH, NEG_SAMPLE_PATH])
call(["mkdir", "-p", POS_SAMPLE_PATH])
call(["mkdir", "-p", NEG_SAMPLE_PATH])

SAMPLE_SIZE = (51, 51)
POS_CRITERIR = 0.8
POS_SAMPLE_PER_FRAME = 5
NEG_SAMPLE_PER_FRAME = 5
EDGE_SAMPLE_PER_FRAME = 0
EDGE_RANGE = 400
MAX_LOOP = 1000
####################################################################

####################### helper functions ###########################
def cropImageAndHistogram(image, bbox, HOG=False):
    # Crop patch and analysis using histogram
    h, w = image.shape[:2]

    crop_x_min = int(max(0, bbox[0] - PATCH_MARGIN))
    crop_x_max = int(min(w - 1, bbox[0] + bbox[2] + PATCH_MARGIN))
    crop_y_min = int(max(0, bbox[1] - PATCH_MARGIN))
    crop_y_max = int(min(h - 1, bbox[1] + bbox[3] + PATCH_MARGIN))

    patch = image[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
    tmp = patch.copy()
    # cv2.normalize(patch, tmp, 0, 255, cv2.NORM_MINMAX) # not good
    if not HOG:
        hist = [cv2.calcHist([tmp], [i], None, [256], [0,256]) for i in range(3)]
    else:
        patch = np.mean(patch, axis=2)
        hist = hog(patch, orientations=9, 
                          pixels_per_cell=(8, 8), 
                          cells_per_block=(3, 3), 
                          block_norm="L2", 
                          visualise=False)
    hist = np.squeeze(hist)

    return hist

def computeHistDist(currHist, prevHist):
    return np.linalg.norm(np.abs(currHist - prevHist))

def drawBox(image, bbox, color=(255, 255, 255)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(image, p1, p2, color, 2, 1)
    return image

def draw_samples(image, pos_bboxs, neg_bboxs):
    for bbox in pos_bboxs:
        image = drawBox(image, bbox, (0, 255, 0))
    for bbox in neg_bboxs:
        image = drawBox(image, bbox, (0, 0, 255))
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
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    return tracker

def create_sample(image, bbox, num_pos=10, num_neg=10, edge_sample=10, pos_criteria=0.8):
    h, w = image.shape[:2]
    def cropImage(image, bbox):
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] >= w or bbox[1] + bbox[3] >= h:
            return None
        bbox = np.array(bbox).astype(int)
        return image[bbox[1]:bbox[1]+bbox[3]+1, bbox[0]:bbox[0]+bbox[2]+1].copy()

    pos_samples, neg_samples = [], []
    pos_bboxs, neg_bboxs = [], []
    bbox = np.array(bbox)

    tmp, centroids = process_bs(image, return_centroids=True)
    if np.mean(tmp) < 1e-6:
        return pos_samples, neg_samples, pos_bboxs, neg_bboxs
    else:
        image = tmp

    # create positive samples
    while len(pos_samples) < num_pos:
        tmp_bbox = bbox.copy()
        tmp_bbox[0] += np.random.randint(-(1 - pos_criteria) * bbox[2], (1 - pos_criteria) * bbox[2])
        tmp_bbox[1] += np.random.randint(-(1 - pos_criteria) * bbox[3], (1 - pos_criteria) * bbox[3])
        patch = cropImage(image, tmp_bbox)
        if patch is not None:
            assert patch.shape[:2] == SAMPLE_SIZE, "{} != {}".format(patch.shape[:2], SAMPLE_SIZE)
            pos_samples.append(patch)
            pos_bboxs.append(tmp_bbox)

    # create negative samples based on negative regions
    bbox_c= np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
    counter = 0
    while len(neg_samples) < num_neg + edge_sample and counter < MAX_LOOP:
        counter += 1
        centroid = centroids[np.random.randint(0, len(centroids))]
        if np.linalg.norm(centroid - bbox_c) < (bbox[2] + bbox[3]) / 2 or centroid[1] < h / 2:
           continue
        rand_x = np.random.randint(-bbox[2]/4, bbox[2]/4)
        rand_y = np.random.randint(-bbox[3]/4, bbox[3]/4)
        tmp_bbox = bbox.copy()
        tmp_bbox[0] = centroid[0]  - bbox[2]/2
        tmp_bbox[1] = centroid[1]  - bbox[3]/2
        patch = cropImage(image, tmp_bbox)
        if patch is not None:
            if np.mean(patch) < 1e-6:
                continue
            assert patch.shape[:2] == SAMPLE_SIZE, "{} != {}".format(patch.shape[:2], SAMPLE_SIZE)
            neg_samples.append(patch)
            neg_bboxs.append(tmp_bbox)

    return pos_samples, neg_samples, pos_bboxs, neg_bboxs

def save_samples(all_pos_samples, all_neg_samples):
    for i, image in enumerate(all_pos_samples):
        cv2.imwrite(os.path.join(POS_SAMPLE_PATH, "pos_%07d.png"%i), image)
    for i, image in enumerate(all_neg_samples):
        cv2.imwrite(os.path.join(NEG_SAMPLE_PATH, "neg_%07d.png"%i), image)

####################################################################

if __name__ == '__main__' :
    # Record variables
    all_pos_samples = []
    all_neg_samples = []

    # Set up tracker.
    sift = SIFT(ROI, TEMPLATE_PATH)
    tracker = creat_tracker(tracker_type)
 
    # Read video
    files = glob.glob(IMAGE_PATH)
    assert len(files) > 0

    _, path_and_file = os.path.splitdrive(files[0])
    path, file = os.path.split(path_and_file)

    video = Video(files, FILE_FORMAT)
    frame_num = video.getFrameNumber()
    frames_counter = 0

    # Record variables
    video_name = path.split('/')[-1] + "_Samples_" + tracker_type + ".mp4"
    video_writer = imageio.get_writer(video_name, fps=15)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
 
    # Use SIFT find init_bbox if init_bbox is none
    if init_bbox is None:
        pt = sift.compute(frame)
        # Stop if both methods failed
        if pt is None:
            raise ValueError("Initial Tracking Failed!!!")
        init_bbox = sift.getBoxFromPt(pt, DEFAULT_BBOX)

    # Initialize tracker with first frame and bounding box
    print("image {} / {}, bbox: {}".format(video.getFrameIdx(), frame_num, init_bbox) )
    ok = tracker.init(frame, init_bbox)

    # Draw initial bbox
    frame = drawBox(frame, init_bbox)

    # Crop patch and analysis using histogram
    prevHist = cropImageAndHistogram(frame, init_bbox, HOG)
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)

        # Crop patch and analysis using histogram
        currHist = cropImageAndHistogram(frame, bbox, HOG)
        dist = computeHistDist(currHist, prevHist)
        prevHist = currHist
        if dist > THRESHOLD_VALUE:
            ok = False

        # Print out current info.
        print ("image {} / {}, bbox: {}, histogram distance: {}".format(video.getFrameIdx(), 
                                                                               frame_num, 
                                                                               bbox,
                                                                               dist) )
 
        if not ok: # Tracking failure
            print ("   %s Failed! Use SIFT!" % tracker_type)
            cv2.putText(frame, "%s Failed! Use SIFT" % tracker_type, (100,300), cv2.FONT_HERSHEY_SIMPLEX, 2.0,(0,0,255),5)
            pt = sift.compute(frame)
            
            # Stop if both methods failed
            if pt is None:
                print ("   Tracking Failed!!!")
                break

            # Update bbox per SIFI point
            if bbox[2] * bbox[3]:
                bbox = sift.getBoxFromPt(pt, bbox)
            else:
                bbox = sift.getBoxFromPt(pt, DEFAULT_BBOX)
            frame = drawBox(frame, bbox)

            # Reinitialize tracker
            del tracker # release the object space
            tracker = creat_tracker(tracker_type)
            tracker.init(frame, bbox) # TODO: This step might have problem after running for a few times
            currHist = cropImageAndHistogram(frame, bbox, HOG)
 
        # Create and record samples
        if frames_counter > INIT_FRAMES_NUM: 
            pos_samples, neg_samples, pos_bboxs, neg_bboxs = create_sample(frame, bbox, 
                                                                           POS_SAMPLE_PER_FRAME, 
                                                                           NEG_SAMPLE_PER_FRAME,
                                                                           EDGE_SAMPLE_PER_FRAME,
                                                                           POS_CRITERIR)
            all_pos_samples += pos_samples
            all_neg_samples += neg_samples

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw current bbox
        frame = drawBox(frame, bbox)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50,170,50),5);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50,170,50), 5);
 
        # Draw samples
        if frames_counter > INIT_FRAMES_NUM: 
            frame = draw_samples(frame, pos_bboxs, neg_bboxs)

        # Display result
        frame_resize = cv2.resize(frame, (500, 500))
        cv2.imshow("Tracking", frame_resize)
        tmp = frame_resize[..., 0].copy()
        frame_resize[..., 0] = frame_resize[..., 2].copy()
        frame_resize[..., 2] = tmp
        video_writer.append_data(frame_resize)
        frames_counter += 1
 
        # Exit if Space pressed
        k = cv2.waitKey(10)
        if k == 32 : break

save_samples(all_pos_samples, all_neg_samples)

print("Finishing... Total image %d, Positive samples: %d, Negative samples: %d" % (frames_counter, len(all_pos_samples), len(all_neg_samples)))
video_writer.close()
