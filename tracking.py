import cv2
import sys
import os
import imageio
import glob
import numpy as np

from sift import SIFT
from video import Video

# Version check
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

############################# Setting ##############################
# Select tracker 
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[1]

# image and template path
IMAGE_PATH = "../images/clear0/*.bmp"
# IMAGE_PATH = "../images/cloudy0/*.bmp"
TEMPLATE_PATH = "templates/kite0/*.png"
# TEMPLATE_PATH = "templates/kite1/*.png"
# File format
# NOTE: Format 0: 2018-1-18-12-49-0-204-original.bmp
#       Format 1: 2017-12-15-10-32-8-595.bmp (without "original")
FILE_FORMAT = 0

# The margin when crop the image for histogram computation
PATCH_MARGIN = 10
THRESHOLD_VALUE = 2000 # TODO: dynamic select this value if possible

# Define an initial bounding box
ROI = [489, 1230, 1407, 609] # The search area when we fail on tracking.
init_bbox = None # None, if no initial bbox
DEFAULT_BBOX = [0, 0, 50, 50] # If init_bbox is none, we use the size of defalt bbox for following tracking
# bbox = (610,  1315, 51, 37) # clear0
# bbox = (1194, 1686, 21, 34) # cloudy0
####################################################################

####################### helper functions ###########################
def cropImageAndHistogram(image, bbox):
    # Crop patch and analysis using histogram
    h, w = image.shape[:2]

    crop_x_min = int(max(0, bbox[0] - PATCH_MARGIN))
    crop_x_max = int(min(w - 1, bbox[0] + bbox[2] + PATCH_MARGIN))
    crop_y_min = int(max(0, bbox[1] - PATCH_MARGIN))
    crop_y_max = int(min(h - 1, bbox[1] + bbox[3] + PATCH_MARGIN))

    patch = image[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
    tmp = patch.copy()
    # cv2.normalize(patch, tmp, 0, 255, cv2.NORM_MINMAX) # not good
    hist = [cv2.calcHist([tmp], [i], None, [256], [0,256]) for i in range(3)]
    hist = np.squeeze(hist)

    return hist

def computeHistDist(currHist, prevHist):
    return np.linalg.norm(np.abs(currHist - prevHist))

def drawBox(image, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(image, p1, p2, (255,255,255), 2, 1)
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
####################################################################

if __name__ == '__main__' :
 
    frames = []

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

    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()
 
    # Use SIFT find init_bbox if init_bbox is none
    if init_bbox is None:
        pt = sift.compute(frame)
        # Stop if both methods failed
        if pt is None:
            raise ValueError("Initial Tracking Failed!!!")
        init_bbox = sift.getBoxFromPt(pt, DEFAULT_BBOX)

    # Initialize tracker with first frame and bounding box
    print "image {} / {}, bbox: {}".format(video.getFrameIdx(), frame_num, init_bbox) 
    ok = tracker.init(frame, init_bbox)

    # Draw initial bbox
    frame = drawBox(frame, init_bbox)

    # Crop patch and analysis using histogram
    prevHist = cropImageAndHistogram(frame, init_bbox)
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
        currHist = cropImageAndHistogram(frame, bbox)
        dist = computeHistDist(currHist, prevHist)
        prevHist = currHist
        if dist > THRESHOLD_VALUE:
            ok = False

        # Print out current info.
        print "image {} / {}, bbox: {}, histogram distance: {}".format(video.getFrameIdx(), 
                                                                       frame_num, 
                                                                       bbox,
                                                                       dist) 
 
        # Draw bounding box
        if ok:
            # Tracking success
            frame = drawBox(frame, bbox)
        else :
            # Tracking failure
            print "   KCF Failed! Use SIFT!"
            cv2.putText(frame, "KCF Failed! Use SIFT", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 2.0,(0,0,255),5)
            pt = sift.compute(frame)
            
            # Stop if both methods failed
            if pt is None:
                print "   Tracking Failed!!!"
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
            currHist = cropImageAndHistogram(frame, bbox)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50,170,50),5);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50,170,50), 5);
 
        # Display result
        frame_resize = cv2.resize(frame, (500, 500))
        cv2.imshow("Tracking", frame_resize)
        tmp = frame_resize[..., 0].copy()
        frame_resize[..., 0] = frame_resize[..., 2].copy()
        frame_resize[..., 2] = tmp
        frames.append(frame_resize)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

print "Finishing... Total image %d" % len(frames)
image_name = path.split('/')[-1] + "_" + tracker_type + ".gif"
print "Save image to {}".format(image_name)
imageio.mimsave(image_name, frames)
