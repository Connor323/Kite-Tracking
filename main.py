#  -*- coding:utf-8 -*-  
from threading import *
import time
from time import sleep

import mmap
import contextlib
import struct
import numbers
import cv2
import numpy as np
import sys
import os
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRACKER_PATH = os.path.join(DIR_PATH, "Kite_Tracking_speedup")
#TRACKER_PATH = os.path.join(DIR_PATH, "Kite_Tracking_angle_detection")

sys.path.insert(0,TRACKER_PATH)
from interface import *
import config as tracker_config

#from Kite_Tracking_angle_detection.interface import *
#import Kite_Tracking_angle_detection.config as tracker_config

"""
# Params for reading image from memory
FULL_MAP_NAME="Local\FramGrapMap"

CAP_PIC_WIDTH=1504
CAP_PIC_HEIGHT=1496

KITE_CROP_WIDTH=36
KITE_CROP_HEIGHT=36

VIEW1_OFFSET=0
VIEW1_SIZE=1024

VIEW2_OFFSET=VIEW1_OFFSET+VIEW1_SIZE
VIEW2_SIZE=KITE_CROP_WIDTH * KITE_CROP_HEIGHT * 4

VIEW3_OFFSET=VIEW2_OFFSET+VIEW2_SIZE
VIEW3_SIZE=CAP_PIC_WIDTH* CAP_PIC_HEIGHT * 4

FULL_MAP_SIZE=VIEW3_OFFSET+VIEW3_SIZE
"""
UPDATE_KERNEL = False

# Number of frames for tracker initialization
INIT_FRAMES_NUM=3

ProgramRunSpeed = 0.04  # Run time for each iteration

READ_FROM_FILE = True # If true, read image from IMAGE_PATH. Otherwise read from camera
DISPLAY=True # If true, display tracking result

ok = False
x = 0 # kite center location x
y = 0 # kite center location y
phi = 0 # kite angle

def read():
    # Called by main thread. Read tracking results.
    return [ok, x, y, phi]

class TrackerThread(Thread):
    def run(self):
        global ok
        global x
        global y
        global phi
        self.ifdo = True
        try:
            self.init_tracker()
            while self.ifdo:
                st_time = time.time()
                # Read frame
                frame = self.get_img(from_file=READ_FROM_FILE)
                
                st_time1 = time.time()
                # Tracking
                ok, bbox, angle, center_loc,_ = self.tracker.update(frame, verbose=False)
                tracktime = time.time() - st_time1

                # Update tracking result
                if ok:
                    x = center_loc[0]
                    y = center_loc[1]
                    phi = angle

                # Display tracking result
                if DISPLAY:
                    if ok:
                        drawBox(frame, bbox)
                        drawAnlge(frame, angle, bbox)
                        drawPoint(frame, center_loc)
                        frame_resize = cv2.resize(frame, (512, 512))
                        cv2.imshow("frame", frame_resize)
                        cv2.waitKey(1)
                    else:
                        frame_resize = cv2.resize(frame, (512, 512))
                        cv2.imshow("frame", frame_resize)
                        cv2.waitKey(1)

                # Switch mode or update kernel if called by main thread
                tracker_config.UPDATE_KERNEL[0] = UPDATE_KERNEL

                runtime = time.time() - st_time
                slp_time = ProgramRunSpeed - (time.time() - st_time) - 0.001
                print("Run time: {:f}ms. Track time: {:f}ms".format(runtime*1000, tracktime*1000))
                if slp_time>0:
                    sleep(slp_time)

        except Exception as ex:  # python 3
            # except Exception , ex:		#python 2
            print(ex)

    def stop(self):
        # print('TrackerThread is stopping ...')
        self.ifdo = False

    def init_tracker(self, init_frames=None):
        # Define and initialize tracker
        print('Initializing tracker...')
        
        self.tracker = Interface()
        if init_frames is None:
            init_frames = []
            for _ in range(INIT_FRAMES_NUM):
                st_time = time.time()
                init_frames.append( self.get_img(from_file=READ_FROM_FILE))
                slp_time = ProgramRunSpeed - (time.time()-st_time) 
                if slp_time>0:
                    sleep( slp_time)
        if len(init_frames)==0:
            print('Cannot find images!')
        self.tracker.init_tracker(init_frames)
        print('Tracker initialized.')    


    def get_img(self, from_file=READ_FROM_FILE):
        # For DEBUG: read image from file
        if from_file == True:
            return self.get_img_from_file()
        """
        # Read camera image from memory
        with contextlib.closing(mmap.mmap(-1, FULL_MAP_SIZE, tagname=FULL_MAP_NAME, access=mmap.ACCESS_READ)) as SharedMemry:
            # Read kite info
            CropPkg = SharedMemry.read(20+36+24)
            [SerialNum, 
            wYear, wMonth, wDayOfWeek, wDay,wHour, wMinute,wSecond,wMilliseconds,
            bFound,Xcoordinate,Ycoordinate,KiteWidth,KiteHeight,KiteRotateAngle,
            Radius,Angle,dConArea,
            KiteImageWidth,KiteImageHeight,KiteImageChannels,OriginalImageWidth,OriginalImageHeight,OriginalImageChannels]= struct.unpack('IHHHHHHHHiffffffffiiiiii', CropPkg)

            # Read original image
            SharedMemry.seek(VIEW3_OFFSET)
            KiteImage=SharedMemry.read(OriginalImageWidth*OriginalImageHeight*OriginalImageChannels)    
            nparr = np.fromstring(KiteImage, np.uint8).reshape((OriginalImageHeight, OriginalImageWidth, OriginalImageChannels )) 
            #nparr = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB) 
        """
        return nparr

    def get_img_from_file(self):
        if not hasattr(self, 'video'):
            files = glob.glob(IMAGE_PATH)
            if len(files) == 0:
                print('Can not find image in path '+ IMAGE_PATH + '\nTracking thread will stop ...')
            assert len(files) > 0
            _, path_and_file = os.path.splitdrive(files[0])
            path, file = os.path.split(path_and_file)
            self.video = Video(files, FILE_FORMAT, START_FRAME)

        ok, frame = self.video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        else:
            return frame

if __name__ == '__main__':

    trackerthread = TrackerThread()
    trackerthread.setDaemon(True)
    trackerthread.start()
    print('TrackerThread is starting ...')
    import datetime 
    while True:
        # currenttime=datetime.datetime.fromtimestamp(time.time())
        # print(currenttime, read())
        sleep(0.04)

    print('I will stop it ...')
    trackerthread.stop()
    trackerthread.join()
