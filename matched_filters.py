import cv2
import imutils
import numpy as np 

from config import *
from utils import *

class MatchedFilter:
	def __init__(self, kernel_path):
		self.kernel = cv2.imread(kernel_path)
		self.kernels, self.angles = self.createMatchedFilterBank()

	def createMatchedFilterBank(self, n=8):
	    '''
	    Given a kernel, create matched filter bank
		
		Params:
			n: number of kernels
	    Return:
	        kernel: [kernel_0, kernel_1, ...]
	    '''


	    K = self.kernel.copy().astype(np.float32)
	    K -= np.mean(K)
	    center = (K.shape[1] / 2, K.shape[0] / 2)
	    cur_rot = 0
	    rotate_interval = 360 / n
	    kernels, angles = [], []

	    for i in range(n):
	        k = imutils.rotate_bound(K, cur_rot)
	        kernels.append(k)
	        angles.append(float(cur_rot))
	        cur_rot += rotate_interval

	    return kernels, angles

	def applyFilters(self, image, bbox):
	    '''
	    Given a filter bank, apply them and record maximum response

	    Params:
	    	image: current frame 
	    Return:
			Selected kernel angle: float value
	    '''
	    patch = cropImage(image, bbox)
	    norm_patch = patch - np.mean(patch)
	    MFR = [cv2.filter2D(norm_patch, -1, k) for k in self.kernels]
	    max_val_mfr = [np.max(mfr) for mfr in MFR]
	    max_idx = np.argmax(max_val_mfr)
	    if DEBUG_MODE:
	    	max_patch = self.kernels[max_idx]
	    	max_patch = (max_patch - np.min(max_patch)) / (np.max(max_patch) - np.min(max_patch))
	    	cv2.imshow("Max Kernel", (max_patch * 255).astype(np.uint8))
	    return self.angles[max_idx]

	def getTargetAngle(self, kernel_angle, bs_patch):
		"""
		Obtain the target angle give the selected kernel angle as a base to avoid aliasing.

		Params:
			kernel_angle: float value
			bs_patch: binary image patch from BS result
		Return:
			object angle: float value
		"""
		_, contours, hierarchy = cv2.findContours(bs_patch, 1, 2)
		max_area = 0
		select_cnt = None
		for cnt in contours:
			M = cv2.moments(cnt)
			if M["m00"] > max_area:
				select_cnt = cnt
				max_area = M["m00"]
		if select_cnt is None:
			return None
		origin_angle = cv2.minAreaRect(select_cnt)[-1]
		angle = kernel_angle - (abs(origin_angle) - 45)
		print angle, kernel_angle, origin_angle
		return angle

		