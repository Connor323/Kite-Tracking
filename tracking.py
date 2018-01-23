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

# The margin when crop the image for histogram computation
PATCH_MARGIN = 10
THRESHOLD_VALUE = 1000

# Define an initial bounding box
ROI = [489, 1230, 1407, 609] # The search area when we fail on tracking.
bbox = (610, 1315, 51, 37) 
# bbox = (1194, 1686, 21, 34) 
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
    hist = [cv2.calcHist([patch], [i], None, [256], [0,256]) for i in range(3)]
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
 
    # Set up tracker.
    sift = SIFT(ROI, TEMPLATE_PATH)
    tracker = creat_tracker(tracker_type)
 
    # Read video
    files = glob.glob(IMAGE_PATH)
    assert len(files) > 0

    _, path_and_file = os.path.splitdrive(files[0])
    path, file = os.path.split(path_and_file)

    video = Video(files)
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
 
    # Initialize tracker with first frame and bounding box
    print "image {} / {}, bbox: {}".format(video.getFrameIdx(), frame_num, bbox) 
    ok = tracker.init(frame, bbox)

    # Crop patch and analysis using histogram
    prevHist = cropImageAndHistogram(frame, bbox)

    frames = []
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
        print "image {} / {}, bbox: {}".format(video.getFrameIdx(), frame_num, bbox) 

        # Crop patch and analysis using histogram
        currHist = cropImageAndHistogram(frame, bbox)
        dist = computeHistDist(currHist, prevHist)
        prevHist = currHist
        if dist > THRESHOLD_VALUE:
            ok = False
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            frame = drawBox(frame, bbox)
        else :
            # Tracking failure
            print "   KCF Failed! Use SIFT!"
            cv2.putText(frame, "KCF Failed! Use SIFT", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            pt = sift.compute(frame)
            
            # Stop if both methods failed
            if pt is None:
                print "   Tracking Failed!!!"
                break
            bbox = sift.getBoxFromPt(pt, bbox)
            frame = drawBox(frame, bbox)

            # Reinitialize tracker
            del tracker
            tracker = creat_tracker(tracker_type)
            tracker.init(frame, bbox) # TODO: This step might have problem after running for a few times
            currHist = cropImageAndHistogram(frame, bbox)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        frame_resize = cv2.resize(frame, (500, 500))
        cv2.imshow("Tracking", frame_resize)
        tmp = frame_resize[..., 0].copy()
        frame_resize[..., 0] = frame_resize[..., 2].copy()
        frame_resize[..., 2] = tmp
        frames.append(frame_resize)
 
        # Exit if ESC pressed
        # cv2.waitKey()
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

print "Finishing... Total image %d" % len(frames)
image_name = path.split('/')[-1] + "_" + tracker_type + ".gif"
print "Save image to {}".format(image_name)
imageio.mimsave(image_name, frames)
