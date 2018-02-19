import cv2
import sys
import os
import imageio
import glob
import copy
import time
import numpy as np

from MLP import MLP_Detection_MP
from video import Video
from matched_filters import MatchedFilter
from utils import * 
from config import *

class Interface:
    def __init__(self, init_bbox=None):
        # Set up tracker.
        self.tracker = creat_tracker(tracker_type)
        # Set up Matched Filter
        self.MF = MatchedFilter(KERNEL_PATH)
        # Initialize variables
        self.prev_angle = None
        self.init_bbox = None
        self.frame_num = 0

    def init_tracker(self, frame, init_bbox=None):
        """
        Initialize tracker given bbox and first frame

        Params: 
            frame: initial frame
            init_bbox: bounding box
        Return:
            ret: if initialization is successful (boolean)
        """
        # Use MLP find init_bbox if init_bbox is none
        if init_bbox is None:
            for _ in range(INIT_FRAMES_NUM):
                MLP_Detection_MP(frame, init_detection=True)
                # Read first frame.
                ok, frame = video.read()
                if not ok:
                    print('Cannot read video file')
                    sys.exit()
            init_bbox, bs_patch = MLP_Detection_MP(frame, init_detection=False)
            # Stop if both methods failed
            if init_bbox is None:
                raise ValueError("Initial Tracking Failed!!!")
            self.init_bbox = copy.copy(init_bbox)

        # Initialize tracker with first frame and bounding box
        return self.tracker.init(frame, init_bbox)

    def update(self, frame, verbose=False):
        """
        Compute bbox and angle given current frame

        Params:
            frame: current color image 
        Return:
            ret: if updating is successful (boolean)
            bbox: bounding bbox
            angle: float value
            center_loc: the center of target [x, y]
        """
        # Start timer
        timer = cv2.getTickCount()

        # Read a new frame
        self.frame_num += 1
        angle = None
        frame_original = frame.copy() # make a copy for result saving
 
        # Update tracker
        ok, bbox = self.tracker.update(frame)
        # bbox limitation (fixed w and h)
        if ok and (tracker_type == "KCF" or bbox[2] * bbox[3] <= 0):
            bbox = list(bbox)
            bbox[2:] = [self.init_bbox[2], self.init_bbox[3]]
            bbox = tuple(bbox)

        if ok:
            # Crop patch and analysis using histogram
            ok, bs_patch = cropImageAndAnalysis(frame, bbox)

        # Use decision buffer to make final decision.
        ok = pushBuffer(ok)
 
        # Draw bounding box
        if not ok:
            # Tracking failure
            bbox, bs_patch = MLP_Detection_MP(frame, init_detection=False)
            if bbox is None:
                if verbose:
                    print("   !!! -> Tracking Failed! Skip current frame...")
                self.prev_angle = None
                return False, None, None, None

            # Reinitialize tracker
            del self.tracker # release the object space
            self.tracker = creat_tracker(tracker_type)
            self.tracker.init(frame, bbox) 
 
        # Apply matched filter to compute the angle of target
        bbox = np.array(bbox).astype(int)
        if bs_patch is not None:
            kernel_angle_idx, center_loc = self.MF.applyFilters(frame_original.copy(), bs_patch.copy(), bbox)
            if kernel_angle_idx is not None:
                angle = self.MF.getTargetAngle(kernel_angle_idx, bs_patch, frame_original.copy(), 
                                               center_loc, bbox, self.prev_angle)
                center_loc = (np.array(center_loc) + np.array(bbox[:2])).astype(int)
                if angle is not None:
                    self.prev_angle = angle
                else:
                    return False, bbox, None, center_loc
            else:
                center_loc = (np.array(center_loc) + np.array(bbox[:2])).astype(int)
                return False, bbox, None, center_loc
        else:
            return False, bbox, None, None
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if verbose:
            # Print out current info.
            print("image {:5d}  |  bbox: {:4d} {:4d} {:3d} {:3d}  |  FPS: {:2d}  |  anlge: {}".format(
                                                                                        self.frame_num, 
                                                                                        int(bbox[0]), int(bbox[1]), 
                                                                                        int(bbox[2]), int(bbox[3]),
                                                                                        int(fps),
                                                                                        angle)) 
        return ok, bbox, angle, center_loc

# This is an example for using Interface
# To avoid opening opencv window and verbose information, 
# please set the variables:
#           WRITE_TMP_RESULT = True
#           DEBUG_MODE = False
# 
if __name__ == "__main__":
    # Read video
    files = glob.glob(IMAGE_PATH)
    assert len(files) > 0

    _, path_and_file = os.path.splitdrive(files[0])
    path, file = os.path.split(path_and_file)

    video = Video(files, FILE_FORMAT, START_FRAME)
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    tracker = Interface()
    tracker.init_tracker(frame)

    while True:
        # Read one frame
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        # Obtain results
        ok, bbox, angle, center_loc = tracker.update(frame, verbose=False)
        if ok:
            print("bbox: {:4d} {:4d} {:3d} {:3d}  |  anlge: {:3d}  |  center: {:4d} {:4d}".format(
                                                                                   int(bbox[0]), int(bbox[1]), 
                                                                                   int(bbox[2]), int(bbox[3]),
                                                                                   int(angle),
                                                                                   center_loc[0], center_loc[1])) 
            drawBox(frame, bbox)
            drawAnlge(frame, angle, bbox)
            drawPoint(frame, center_loc)
            frame_resize = cv2.resize(frame, (512, 512))
            cv2.imshow("frame", frame_resize)
            cv2.waitKey(1)
        else:
            print("   ->Tracking failed!!!")








