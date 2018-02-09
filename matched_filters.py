import cv2
import threading
import numpy as np 

from config import *
from utils import *

class MatchedFilter:
    def __init__(self, kernel_path):
        self.kernel = cv2.imread(kernel_path)
        self.kernels, self.angles = self.createMatchedFilterBank(-90)

    def createMatchedFilterBank(self, init_angle=0, prev_kernels=None, prev_angles=None):
        '''
        Given a kernel, create matched filter bank with different rotation angles

        Params:
            init_angle: the target angle 
            prev_kernels:
            prev_angles:  
        Return:
            kernels: [[kernel_0, kernel_1, ...], [kernel_0, kernel_1, ...]]
            angles: [a0, a1, ...]
        '''
        def rotate_bound(image, angle):
            # grab the dimensions of the image and then determine the
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

        def bufferPop(kernels, angles):
            if USE_UPDATE_BUFFER:
                if len(kernels) > NUM_ROTATION * KERNAL_UPDATE_FREQ:
                    kernels = kernels[NUM_ROTATION:]
                if len(angles) > NUM_ROTATION * KERNAL_UPDATE_FREQ:
                    angles = angles[NUM_ROTATION:]
            return kernels, angles

        K = self.kernel.copy().astype(np.float32)
        K = cv2.normalize(K, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        K -= np.mean(K)
        cur_rot = 0.
        rotate_interval = 360 / NUM_ROTATION
        if prev_kernels is not None:
            kernels, angles = [] + prev_kernels, [] + prev_angles
        else:
            kernels, angles = [], []

        for i in range(NUM_ROTATION):
            k = rotate_bound(K.copy(), cur_rot)
            k -= np.mean(k)
            kernels.append(k)
            angles.append(self.clip_angle(cur_rot + init_angle))
            cur_rot += rotate_interval

        return bufferPop(kernels, angles)

    def clip_angle(self, angle):
        """
        Clip the angle to the range of 0 - 359.

        Params:
            angle: float value
        Return:
            angle in [0, 360)
        """
        return angle % 360

    def applyFilters(self, image, bs_patch, bbox):
        '''
        Given a filter bank, apply them and record maximum response

        Params:
            image: current frame 
            bs_patch: BS patch result
        Return:
            Selected kernel angle idx: int value
        '''
        def work(patch, thread, MFR):
            for i in range(thread, len(self.kernels), NUM_THREADS):
                MFR[i] = np.max(cv2.filter2D(norm_patch, -1, self.kernels[i], borderType=cv2.BORDER_CONSTANT))

        def MFR_MP(patch):
            # Assign jobs
            threads = []
            MFR = [[] for _ in range(len(self.kernels))]
            for thread in range(NUM_THREADS):
                t = threading.Thread(target=work, args=(patch, thread, MFR))
                t.start()
                threads.append(t)

            # Wait for computing
            still_alive = True
            while still_alive:
                still_alive = False
                for t in threads:
                    if t.isAlive():
                        still_alive = True

            return MFR

        def getKernelIdx(max_per_MFR):
            max_per_step, max_idx_per_step, selected_angle_per_step = [], [], []
            for i in range(len(self.kernels) / NUM_ROTATION):
                max_per_step.append(np.max(max_per_MFR[i * NUM_ROTATION : (i+1) * NUM_ROTATION]))
                max_idx_per_step.append(NUM_ROTATION*i + np.argmax(max_per_MFR[i * NUM_ROTATION : (i+1) * NUM_ROTATION]))
                selected_angle_per_step.append(int(self.angles[max_idx_per_step[-1]] / (360 / NUM_ROTATION)))
            max_per_step = np.array(max_per_step)
            max_idx_per_step = np.array(max_idx_per_step)
            selected_angle_per_step = np.array(selected_angle_per_step)

            angles, counts = np.unique(selected_angle_per_step, return_counts=True)
            max_counts = np.max(counts)
            selected_angles = angles[counts == max_counts]

            tmp_idx = np.zeros(len(self.kernels) / NUM_ROTATION)
            for selected_angle in selected_angles:
                tmp_idx[selected_angle_per_step == selected_angle] = 1
            tmp_idx = tmp_idx.astype(bool)

            max_indices = max_idx_per_step[tmp_idx]
            max_idx_kernel = max_indices[np.argmax(max_per_step[tmp_idx])]
            return max_idx_kernel

        patch = cropImage(image, bbox)
        if patch is None:
            return None

        patch = cv2.normalize(patch, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_patch = patch - np.mean(patch)
        max_per_MFR = MFR_MP(norm_patch)
        # MFR = [cv2.matchTemplate(norm_patch.copy(), k.copy(), TEMPLATE_MATCHING_MATHOD, mask=(k != 0).astype(np.float32)) \
                                  # for k in self.kernels]
        max_idx_kernel = getKernelIdx(max_per_MFR)

        if DEBUG_MODE:
            max_patch = self.kernels[max_idx_kernel]
            zero_mask = max_patch == 0
            max_patch = (max_patch - np.min(max_patch)) / (np.max(max_patch) - np.min(max_patch))
            norm_patch = (norm_patch - np.min(norm_patch)) / (np.max(norm_patch) - np.min(norm_patch))
            max_patch[zero_mask] = 0
            KERNEL_RECORD[0] = (max_patch * 255).astype(np.uint8)
            PATCH_RECORD[0] = (norm_patch * 255).astype(np.uint8)
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

        def clip_index(idx):
            return idx % len(self.angles)

        def anglesDistance(angle1, angle2):
            return min(self.clip_angle(angle1 - angle2), 
                       self.clip_angle(angle2 - angle1))

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
        origin_angle = self.clip_angle(origin_angle)
        kernel_angle = self.angles[kernel_angle_idx]

        if anglesDistance(kernel_angle, origin_angle) > THRESH_ANGLE_FLIP:
            angle = origin_angle + 180
        else:
            angle = origin_angle

        return self.clip_angle(angle)

    def updateKernel(self, image, bs_patch, bbox, angle):
        """
        Update the kernel based on the current BS result and bbox

        Params:
            kernel_angle_idx: int value
            bs_patch: binary image patch from BS result
            bbox: bounding box
            angle: float value of current target angle
        """
        patch = cropImage(image, bbox)
        if patch is None:
            return

        _, contours, hierarchy = cv2.findContours(bs_patch, 1, 2)
        max_area = 0
        select_cnt = None
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > max_area:
                select_cnt = cnt
                max_area = M["m00"]
        if select_cnt is None:
            return
        bbox_tight = cv2.boundingRect(select_cnt)
        patch_tight = cropImage(patch, bbox_tight)
        if patch_tight is not None:
            self.kernel = patch_tight.astype(np.float32)
            self.kernels, self.angles = self.createMatchedFilterBank(angle, self.kernels, self.angles)







        