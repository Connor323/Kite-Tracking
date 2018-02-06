import cv2
import numpy as np 

from config import *
from utils import *

class MatchedFilter:
    def __init__(self, kernel_path):
        self.kernel = cv2.imread(kernel_path)
        self.kernels, self.angles = self.createMatchedFilterBank()

    def createMatchedFilterBank(self):
        '''
        Given a kernel, create matched filter bank with different rotation angle and scaling

        Return:
            kernel: [[kernel_0, kernel_1, ...], [kernel_0, kernel_1, ...]]
        '''
        def getPyramidKernels(kernel):
            res = []
            for order in range(-NUM_SCALING/2, NUM_SCALING/2):
                scaling = SCALING_RATIO ** order
                size = (int(kernel.shape[1] * scaling), int(kernel.shape[0] * scaling))
                resize_k = cv2.resize(kernel, size)
                resize_k -= np.mean(resize_k)
                res.append(resize_k)
            return res

        def rotate_bound(image, angle):
            # grab the dimensions of the image and then determine the
            # center
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            # grab the rotation matrix (applying the negative of the
            # angle to rotate clockwise), then grab the sine and cosine
            # (i.e., the rotation components of the matrix)
            M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            # perform the actual rotation and return the image
            return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        n = 8 # number of rotations
        K = self.kernel.copy().astype(np.float32)
        K -= np.mean(K)
        center = (K.shape[1] / 2, K.shape[0] / 2)
        cur_rot = 0
        rotate_interval = 360 / n
        kernels, angles = [], []

        for i in range(n):
            k = rotate_bound(K, cur_rot)
            kernels.append(getPyramidKernels(k))
            angles.append(float(cur_rot))
            cur_rot += rotate_interval
        return kernels, angles

    def applyFilters(self, image, bbox):
        '''
        Given a filter bank, apply them and record maximum response

        Params:
            image: current frame 
        Return:
            Selected kernel angle idx: int value
        '''
        patch = cropImage(image, bbox)
        if patch is None:
            return None
            
        norm_patch = patch - np.mean(patch)
        MFR = [cv2.filter2D(norm_patch, -1, k, borderType=cv2.BORDER_CONSTANT) for kernels in self.kernels for k in kernels]
        max_val_mfr = [np.max(mfr) for mfr in MFR]
        max_idx = np.argmax(max_val_mfr)
        max_idx_kernel, max_idx_pyramid = max_idx / NUM_SCALING, max_idx % NUM_SCALING
        if DEBUG_MODE:
            max_patch = self.kernels[max_idx_kernel][max_idx_pyramid]
            max_patch = (max_patch - np.min(max_patch)) / (np.max(max_patch) - np.min(max_patch))
            KERNEL_RECORD[0] = (max_patch * 255).astype(np.uint8)
        return max_idx_kernel

    def getTargetAngle(self, kernel_angle_idx, bs_patch):
        """
        Obtain the target angle give the selected kernel angle as a base to avoid aliasing.

        Params:
            kernel_angle_idx: int value
            bs_patch: binary image patch from BS result
        Return:
            object angle: float value
        """
        def dist(pt1, pt2):
            return np.linalg.norm(np.array(pt1) - np.array(pt2))

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
        rect = cv2.minAreaRect(select_cnt)
        bbox = cv2.boxPoints(rect)
        d1 = dist(bbox[0], bbox[1])
        d2 = dist(bbox[1], bbox[2])
        if d1 > d2:
            if abs(bbox[1][0] - bbox[2][0]) < 1e-10:
                slide = np.inf if bbox[1][1] - bbox[2][1] > 0 else -np.inf
                origin_angle_radian = np.arctan(slide)
            else:
                origin_angle_radian = np.arctan(float(bbox[1][1] - bbox[2][1]) / (bbox[1][0] - bbox[2][0]))
            origin_angle = origin_angle_radian / np.pi * 180
        else:
            if abs(bbox[0][0] - bbox[1][0]) < 1e-10:
                slide = np.inf if bbox[0][1] - bbox[1][1] > 0 else -np.inf
                origin_angle_radian = np.arctan(slide)
            else:
                origin_angle_radian = np.arctan(float(bbox[0][1] - bbox[1][1]) / (bbox[0][0] - bbox[1][0]))
            origin_angle = origin_angle_radian / np.pi * 180

        if kernel_angle_idx == 0: # 0 degree
            if origin_angle <= 0:
                angle = origin_angle + 360
            else:
                angle = origin_angle + 180
        elif kernel_angle_idx == 1: # 45 degree
            if origin_angle <= 0:
                angle = origin_angle + 360
            else:
                if origin_angle < 45:
                    angle = origin_angle
                else:
                    angle = origin_angle + 180
        elif kernel_angle_idx == 2: # 90 degree
            if origin_angle <= 0:
                angle = origin_angle + 360
            else:
                angle = origin_angle
        elif kernel_angle_idx == 3: # 135 degree
            if origin_angle >= 0:
                angle = origin_angle
            else:
                angle = origin_angle + 360
        elif kernel_angle_idx == 4: # 180 degree
            if origin_angle <= 0:
                angle = origin_angle + 180
            else:
                angle = origin_angle
        elif kernel_angle_idx == 5: # 225 degree
            if origin_angle <= 0:
                angle = origin_angle + 180
            else:
                if origin_angle < 45:
                    angle = origin_angle + 180
                else:
                    angle = origin_angle
        elif kernel_angle_idx == 6: # 270 degree
            angle = origin_angle + 180
        elif kernel_angle_idx == 7: # 315 degree
            if origin_angle >= 0:
                angle = origin_angle + 180
            else:
                if origin_angle < -45:
                    angle = origin_angle + 360
                else:
                    angle = origin_angle + 180
        return angle

        