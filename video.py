import cv2
import os
import numpy as np

# NOTE: Format 0: 2018-1-18-12-49-0-204-original.bmp
#       Format 1: 2017-12-15-10-32-8-595.bmp (without "original")

class Video:
    """Video reader classs"""
    def __init__(self, files, file_format=0, start_frame=None):
        """
        Params:
            files: the list of frame names including path
            file_format: Format 0 / 1; See the note above.
            start_frame: the name of start frame (in case we want start in the middel of video)
        """
        assert files is not None

        self.counter = -1
        self.file_format = file_format
        self.files = self.sort(files)
        self.start_idx = 0
        if start_frame is not None:
            self.start_idx = np.squeeze(np.where(self.files == start_frame))
        print("Starting from frame: %d" % (self.start_idx))
        self.files = self.files[self.start_idx:]   
        self.iter = iter(self.files)
    
    def sort(self, files):
        """
        Sort the file names based on the naming protocal (this might need to modify if we change the name protocal)
        """
        idx = []
        for file in files:
            file = file.split("/")[-1].split("-")
            if self.file_format == 0:
                tmp = file[-2] 
                idx.append([int(file[-6]), int(file[-5]), int(file[-4]), int(file[-3]), int(tmp)]) 
            elif self.file_format == 1:
                tmp = file[-1].split(".")[0] 
                idx.append([int(file[-2]), int(tmp)]) 
            else:
                raise ValueError("Unrecongnized format (available 0/1): {}".format(self.file_format))
        idx = sorted(range(len(idx)), key=lambda k: idx[k])
        files = np.array(files)
        return files[idx]

    def isOpened(self):
        """
        Check if file is been loaded. 
        """
        return len(self.files) > 0

    def read(self):
        """
        Read one frame.
        Return:
            ret: if is reading successful
            image: current frame
        """
        try:
            file = next(self.iter)
            image = cv2.imread(file)
            self.counter += 1
        except(StopIteration):
            return False, None
        return True, image

    def getFrameNumber(self):
        """
        Get the total numbe of file (the number counts from the start frame).
        """
        return len(self.files)

    def getFrameName(self):
        """
        Get the current image file name
        """
        file = self.files[self.counter]
        return os.path.basename(file)

    def getFrameIdx(self):
        """
        Get current frame index.
        """
        return self.counter

    def release(self):
        """
        For consistency of OpenCV video reader. 
        """
        pass