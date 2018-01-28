import cv2
import numpy as np

# NOTE: Format 1: 2018-1-18-12-49-0-204-original.bmp
#       Format 2: 2017-12-15-10-32-8-595.bmp (without "original")

class Video:
    """docstring for Video"""
    def __init__(self, files, file_format=0, start_frame=None):
        assert files is not None
        self.counter = 0
        self.file_format = file_format
        self.files = self.sort(files)
        self.start_idx = None
        try:
            self.start_idx = np.squeeze(np.where(self.files == start_frame))
            if type(self.start_idx) != type(0):
                self.start_idx = 0
        except ValueError:
            self.start_idx = 0
        print "Starting from frame: %d" % (self.start_idx)
        self.files = self.files[self.start_idx:]   
        self.iter = iter(self.files)
    
    def sort(self, files):
        idx = []
        for file in files:
            file = file.split("/")[-1].split("-")
            if self.file_format == 0:
                tmp = file[-2] 
                idx.append([int(file[-4]), int(file[-3]), int(tmp)]) 
            elif self.file_format == 1:
                tmp = file[-1].split(".")[0] 
                idx.append([int(file[-2]), int(tmp)]) 
            else:
                raise ValueError("Unrecongnized format (available 0/1): {}".format(self.file_format))
        idx = sorted(range(len(idx)), key=lambda k: idx[k])
        files = np.array(files)
        return files[idx]

    def isOpened(self):
        return len(self.files) > 0

    def read(self):
        try:
            file = self.iter.next()
            image = cv2.imread(file)
            self.counter += 1
        except StopIteration, e:
            return False, None
        return True, image

    def getFrameNumber(self):
        return len(self.files)

    def getFrameIdx(self):
        return self.counter

    def release(self):
        pass