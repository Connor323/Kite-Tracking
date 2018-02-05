import cv2
import sys
import os
import imageio
import glob
import numpy as np

from MLP import MLP_Detection_MP
from video import Video
from matched_filters import MatchedFilter
from utils import * 
from config import *

if __name__ == '__main__' :
    # Set up tracker.
    tracker = creat_tracker(tracker_type)

    # Set up Matched Filter
    MF = MatchedFilter(KERNEL_PATH)
 
    # Read video
    files = glob.glob(IMAGE_PATH)
    assert len(files) > 0

    _, path_and_file = os.path.splitdrive(files[0])
    path, file = os.path.split(path_and_file)

    video = Video(files, FILE_FORMAT, START_FRAME)
    frame_num = video.getFrameNumber()

    # Record variables
    image_name = path.split('/')[-1] + "_" + tracker_type + ".mp4"
    video_writer = imageio.get_writer(image_name, fps=RECORD_FPS)
    frames_counter = 0

    # Create handler when press Ctrl + C
    signal.signal(signal.SIGINT, signal_handler)

    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()
 
    # Use MLP find init_bbox if init_bbox is none
    if init_bbox is None:
        init_bbox, bs_patch = MLP_Detection_MP(frame, init_detection=True)
        # Stop if both methods failed
        if init_bbox is None:
            raise ValueError("Initial Tracking Failed!!!")

    # Initialize tracker with first frame and bounding box
    print "image {} / {}, initial bbox: {}".format(video.getFrameIdx(), frame_num, init_bbox) 
    ok = tracker.init(frame, init_bbox)

    # Draw initial bbox
    frame = drawBox(frame, init_bbox)

    while True:
        # Read a new frame
        read_ok, frame = video.read()
        if not read_ok:
            break
        frame_original = frame.copy() # make a copy for result saving
         
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
            ok, bs_patch = cropImageAndAnalysis(clf, frame, bbox)

        # Use decision buffer to make final decision.
        ok = pushBuffer(ok)
        
        # Print out current info.
        print "image {} / {}, bbox: {}".format(video.getFrameIdx(), frame_num, np.array(bbox).astype(int)) 
 
        # Draw bounding box
        if ok:
            # Tracking success
            frame = drawBox(frame, bbox)
        else :
            # Tracking failure
            if DEBUG_MODE:
                print "   %s Failed! Use classifier!" % tracker_type
            bbox, bs_patch = MLP_Detection_MP(frame, init_detection=False)
            if bbox is None:
                print "   !!! -> Tracking Failed! Skip current frame..."
                cv2.putText(frame, "Tracking Failed! Skip current frame...", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 2.0,(0,0,255),5)
                video_writer.append_data(displayFrame(frame, frame_original, bbox, video))
                frames_counter += 1
                if not WRITE_TMP_RESULT:
                    # Exit if Space pressed
                    k = cv2.waitKey(10)
                    if k == 32 : break
                continue
            
            # Draw bbox
            frame = drawBox(frame, bbox)
            # Reinitialize tracker
            del tracker # release the object space
            tracker = creat_tracker(tracker_type)
            tracker.init(frame, bbox) 
 
        # Apply matched filter to compute the angle of target
        if bs_patch is not None:
            kernel_angle = MF.applyFilters(frame_original, bbox)
            angle = MF.getTargetAngle(kernel_angle, bs_patch)
            drawAnlge(frame, angle, bbox)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50,170,50), 5);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50,170,50), 5);
 
        # Display result
        video_writer.append_data(displayFrame(frame, frame_original, bbox, video))
        frames_counter += 1

        if not WRITE_TMP_RESULT:
            # Exit if Space pressed
            k = cv2.waitKey()
            # k = cv2.waitKey(10)
            if k == 32 : break

print "Finishing... Total image %d" % frames_counter
print "Save image to {}".format(image_name)
video_writer.close()

