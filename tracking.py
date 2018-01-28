import cv2
import sys
import os
import imageio
import glob
import numpy as np
from sklearn.externals import joblib

from sift import SIFT
from MLP import MLP_Detection, MLP_Detection_MP
from video import Video
from utils import * 
from config import *

if __name__ == '__main__' :
    # Set up tracker.
    if not USE_CLF:
        sift = SIFT(ROI, TEMPLATE_PATH)
    tracker = creat_tracker(tracker_type)
 
    # Read video
    files = glob.glob(IMAGE_PATH)
    assert len(files) > 0

    _, path_and_file = os.path.splitdrive(files[0])
    path, file = os.path.split(path_and_file)

    video = Video(files, FILE_FORMAT, START_FRAME)
    frame_num = video.getFrameNumber()

    # Record variables
    image_name = path.split('/')[-1] + "_" + tracker_type + ".mp4"
    video_writer = imageio.get_writer(image_name, fps=15)
    frames_counter = 0

    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()
 
    # Use SIFT/MLP find init_bbox if init_bbox is none
    if init_bbox is None:
        if not USE_CLF: 
            pt = sift.compute(frame)
            # Stop if both methods failed
            if pt is None:
                raise ValueError("Initial Tracking Failed!!!")
            init_bbox = sift.getBoxFromPt(pt, DEFAULT_BBOX)
        else:
            if DO_MP:
                init_bbox = MLP_Detection_MP(frame, clf, NUM_THREADS, PROB_CRITERIA, step_size=STEP_SIZE, record_size=RECORD_SIZE)
            else:
                init_bbox = MLP_Detection(frame, clf, PROB_CRITERIA, step_size=STEP_SIZE, record_size=RECORD_SIZE)
            # Stop if both methods failed
            if init_bbox is None:
                raise ValueError("Initial Tracking Failed!!!")

    # Initialize tracker with first frame and bounding box
    print "image {} / {}, initial bbox: {}".format(video.getFrameIdx(), frame_num, init_bbox) 
    ok = tracker.init(frame, init_bbox)

    # Draw initial bbox
    frame = drawBox(frame, init_bbox)

    # Crop patch and analysis using histogram
    hist = cropImageAndHistogram(frame, init_bbox, HOG)
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)

        # bbox limitation (fixed w and h)
        if ok and (tracker_type == "KCF" or bbox[2] * bbox[3] <= 0):
            bbox = list(bbox)
            bbox[2:] = [init_bbox[2], init_bbox[3]]
            bbox = tuple(bbox)

        if ok:
            # Crop patch and analysis using histogram
            ok, hist, dist = cropImageAndAnalysis(clf, frame, bbox, hist, HOG, USE_CLF)

        # Print out current info.
        print "image {} / {}, bbox: {}, feature distance: {}".format(video.getFrameIdx(), 
                                                                     frame_num, 
                                                                     bbox, 
                                                                     dist) 
 
        # Draw bounding box
        if ok:
            # Tracking success
            frame = drawBox(frame, bbox)
        else :
            # Tracking failure
            print "   %s Failed! Use classifier!" % tracker_type
            if not USE_CLF: # Use SIFT
                pt = sift.compute(frame)
                # Stop if both methods failed
                if pt is None:
                    print "   Tracking Failed! Skip current frame..."
                    cv2.putText(frame, "Tracking Failed! Skip current frame...", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 2.0,(0,0,255),5)
                    video_writer.append_data(displayFrame(frame, RECORD_SIZE))
                    frames_counter += 1
                    # Exit if Space pressed
                    k = cv2.waitKey(10)
                    if k == 32 : break
                    continue
                # Update bbox per SIFI point
                if bbox[2] * bbox[3]:
                    bbox = sift.getBoxFromPt(pt, bbox)
                else:
                    bbox = sift.getBoxFromPt(pt, DEFAULT_BBOX)
            else: # Use HOG + MLP
                if DO_MP:
                    bbox = MLP_Detection_MP(frame, clf, NUM_THREADS, PROB_CRITERIA, step_size=STEP_SIZE, record_size=RECORD_SIZE)
                else:
                    bbox = MLP_Detection(frame, clf, PROB_CRITERIA, step_size=STEP_SIZE, record_size=RECORD_SIZE)
                if bbox is None:
                    print "   Tracking Failed! Skip current frame..."
                    cv2.putText(frame, "Tracking Failed! Skip current frame...", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 2.0,(0,0,255),5)
                    video_writer.append_data(displayFrame(frame, RECORD_SIZE))
                    frames_counter += 1
                    # Exit if Space pressed
                    k = cv2.waitKey(10)
                    if k == 32 : break
                    continue
            
            # Draw bbox
            frame = drawBox(frame, bbox)
            # Reinitialize tracker
            del tracker # release the object space
            # tracker.clear()
            tracker = creat_tracker(tracker_type)
            tracker.init(frame, bbox) # TODO: This step might have problem after running for a few times
            hist = cropImageAndHistogram(frame, bbox, HOG)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50,170,50), 5);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50,170,50), 5);
 
        # Display result
        video_writer.append_data(displayFrame(frame, RECORD_SIZE))
        frames_counter += 1

        # Exit if Space pressed
        k = cv2.waitKey(10)
        if k == 32 : break

print "Finishing... Total image %d" % frames_counter
print "Save image to {}".format(image_name)
video_writer.close()