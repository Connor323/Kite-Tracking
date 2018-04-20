import cv2
import threading
import numpy as np 
import os

from config import *
from utils import *

class MatchedFilter:
    def __init__(self, kernel_path):
        self.kernel_dir = os.path.dirname(kernel_path)
        self.kernel_idx = int(os.path.basename(kernel_path).split("_")[1].split(".")[0])
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
            M[0, 2] += (nW // 2) - cX
            M[1, 2] += (nH // 2) - cY

            # perform the actual rotation and return the image
            return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        K = self.kernel.copy()
        K = cv2.normalize(K, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        K -= np.mean(K)
        cur_rot = 0.
        rotate_interval = 360 // NUM_ROTATION
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

        return kernels, angles

    def clip_angle(self, angle):
        """
        Clip the angle to the range of 0 - 359.

        Params:
            angle: float value
        Return:
            angle in [0, 360)
        """
        return angle % 360

    def findTightBboxFromBS(self, bs_patch):
        """
        Find the tight bbox from the BS binary image

        Params:
            bs_patch: the binary image patch/image from BS result
        Return:
            tight bbox
            tight rect
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
            return None, None

        rect = cv2.minAreaRect(select_cnt)
        bbox = cv2.boxPoints(rect)
        return bbox, rect

    def findBboxFromBS(self, bs_patch):
        """
        Find the bbox from the BS binary image

        Params:
            bs_patch: the binary image patch/image from BS result
        Return:
            bbox
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
        else:
            x,y,w,h = cv2.boundingRect(select_cnt)
            return np.array([x,y,w,h])

    def angles_distance(self, angle1, angle2):
        """
        Compute the distance between two angles

        Params:     
            angle1, angle2
        Return: 
            angle distance in [0, 360)
        """
        diff = (angle1 - angle2) / 180 * np.pi
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return abs(diff / np.pi * 180)
        # return np.min([self.clip_angle(angle1 - angle2), self.clip_angle(angle2 - angle1)])

    def angles_difference(self, angle1, angle2):
        """
        Compute the difference between two angles

        Params:     
            angle1, angle2
        Return: 
            angle difference in [-180, 180]
        """
        diff = (angle1 - angle2) / 180 * np.pi
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return diff / np.pi * 180

    @staticmethod
    def update_kernel_status():
        """
        Obtian the current update_kernel status from keyboard

        Params:
            None
        Return:
            None
        """
        k = cv2.waitKey(1)
        if k == ord('a'):
            UPDATE_KERNEL[0] = True
        else:
            UPDATE_KERNEL[0] = False

    def saveNewKernel(self):
        """
        Save the new kernel in the previous kernel directory

        Params:
            None
        Return:
            None
        """
        self.kernel_idx += 1
        cv2.imwrite(os.path.join(self.kernel_dir, 
                               "kernel_" + str(self.kernel_idx) + ".bmp"), 
                  self.kernel)

    def applyFilters(self, image, bs_patch, bbox):
        '''
        Given a filter bank, apply them and record maximum response

        Params:
            image: current frame 
            bs_patch: BS patch result
        Return:
            Selected kernel angle idx: int value
        '''
        def work(patch, thread, MFR, LOC):
            for i in range(thread, len(self.kernels), NUM_THREADS_MFR):
                res = cv2.filter2D(norm_patch, -1, self.kernels[i], borderType=cv2.BORDER_CONSTANT)
                res = np.mean(res, axis=2)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                MFR[i] = max_val
                LOC[i] = max_loc

        def MFR_MP(patch):
            # Assign jobs
            threads = []
            MFR = [[] for _ in range(len(self.kernels))]
            LOC = [[] for _ in range(len(self.kernels))]
            for thread in range(NUM_THREADS_MFR):
                t = threading.Thread(target=work, args=(patch, thread, MFR, LOC))
                t.start()
                threads.append(t)

            # Wait for computing
            still_alive = True
            while still_alive:
                still_alive = False
                for t in threads:
                    if t.isAlive():
                        still_alive = True

            return MFR, LOC

        patch_original = cropImage(image, bbox)
        if patch_original is None:
            return None, None

        patch = cv2.normalize(patch_original, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_patch = patch - np.mean(patch)
        max_per_MFR, LOC = MFR_MP(norm_patch)
        max_idx_kernel = np.argmax(max_per_MFR) 
        max_location = LOC[max_idx_kernel]

        self.update_kernel_status()
        if UPDATE_KERNEL[0]:
            tmp_bbox = self.findBboxFromBS(bs_patch)
            if tmp_bbox is not None: 
                tmp_kernel = cropImage(patch_original, tmp_bbox)
                if tmp_kernel is not None: 
                    print("   -> update kernel")
                    self.kernel = tmp_kernel 
                    self.kernels, self.angles = self.createMatchedFilterBank(self.angles[max_idx_kernel])
                    self.saveNewKernel()

        if DEBUG_MODE:
            max_kernel_patch = self.kernels[max_idx_kernel]
            max_kernel_patch = (max_kernel_patch - np.min(max_kernel_patch)) / (np.max(max_kernel_patch) - np.min(max_kernel_patch))
            norm_patch = (norm_patch - np.min(norm_patch)) / (np.max(norm_patch) - np.min(norm_patch))
            cv2.circle(norm_patch, tuple(max_location), 1, (0, 255, 0), -1)
            KERNEL_RECORD[0] = (max_kernel_patch * 255).astype(np.uint8)
            PATCH_RECORD[0] = (norm_patch * 255).astype(np.uint8)
        return max_idx_kernel, max_location

    def getTargetAngle(self, kernel_angle_idx, bs_patch, image, max_loc, bbox, prev_angle):
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

        def getMeanValueFromArea(patch):
            s_patch = patch[..., 1].astype(np.uint8)
            ret2, s_patch = cv2.threshold(s_patch, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
            if s_patch is None: # TODO: check why is None
                return 0
            s_area = s_patch[s_patch > 0]
            if s_area.size:
                return np.mean(s_area)
            else:
                return 0

        def findColorLoc(patch, bbox):
            bbox = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1]), np.max(bbox[:, 0]), np.max(bbox[:, 1])])
            bbox[2:] = bbox[2:] - bbox[:2]
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox = bbox.astype(int)
            if bbox[2] < bbox[3]:
                area1 = patch[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]//2] # left: 180
                area1 = getMeanValueFromArea(area1)
                area2 = patch[bbox[1]:bbox[1]+bbox[3], bbox[0]+bbox[2]//2:bbox[0]+bbox[2]] # right: 0
                area2 = getMeanValueFromArea(area2)
                angles = [180, 0]
            else:
                area1 = patch[bbox[1]:bbox[1]+bbox[3]//2, bbox[0]:bbox[0]+bbox[2]] # upper: 270
                area1 = getMeanValueFromArea(area1)
                area2 = patch[bbox[1]+bbox[3]//2:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] # bottom: 90
                area2 = getMeanValueFromArea(area2)
                angles = [270, 90]
            return angles[np.argmax([area1, area2])]

        def refineAngle(patch, bbox, bbox_rect, max_loc):
            max_loc = np.array(max_loc)
            patch_h, patch_w = patch.shape[:2]

            _, (w, h), angle = bbox_rect
            old_c = np.array([np.mean(bbox[:, 0]), np.mean(bbox[:, 1])])
            tmp = np.ones([4, 3], float)
            bbox = bbox + (max_loc - old_c)
            tmp[:, :2] = bbox
        
            M = cv2.getRotationMatrix2D(tuple(max_loc), angle, 1)
            rotated_patch = cv2.warpAffine(patch, M, (patch_w, patch_h))
            bbox = M.dot(tmp.T).T
            selected_angle = findColorLoc(rotated_patch, bbox)

            return self.clip_angle(selected_angle + angle)


        tight_bbox, tight_rect = self.findTightBboxFromBS(bs_patch)
        if tight_bbox is None:
            return None

        d1 = dist(tight_bbox[0], tight_bbox[1])
        d2 = dist(tight_bbox[1], tight_bbox[2])
        if d1 > d2:
            if abs(tight_bbox[1][0] - tight_bbox[2][0]) < 1e-10:
                slide = np.inf if tight_bbox[1][1] - tight_bbox[2][1] > 0 else -np.inf
                origin_angle_radian = np.arctan(slide)
            else:
                origin_angle_radian = np.arctan(float(tight_bbox[1][1] - tight_bbox[2][1]) / (tight_bbox[1][0] - tight_bbox[2][0]))
            origin_angle = origin_angle_radian / np.pi * 180
        else:
            if abs(tight_bbox[0][0] - tight_bbox[1][0]) < 1e-10:
                slide = np.inf if tight_bbox[0][1] - tight_bbox[1][1] > 0 else -np.inf
                origin_angle_radian = np.arctan(slide)
            else:
                origin_angle_radian = np.arctan(float(tight_bbox[0][1] - tight_bbox[1][1]) / (tight_bbox[0][0] - tight_bbox[1][0]))
            origin_angle = origin_angle_radian / np.pi * 180
        origin_angle = self.clip_angle(origin_angle)
        kernel_angle = self.angles[kernel_angle_idx]

        if self.angles_distance(kernel_angle, origin_angle) > THRESH_ANGLE_DISTANCE:
            angle0 = self.clip_angle(origin_angle + 180)
        else:
            angle0 = origin_angle

        patch = cropImage(image, bbox)
        patch[bs_patch == 0] = 0
        if patch is None:
            return None
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        angle1 = self.clip_angle(refineAngle(patch, tight_bbox, tight_rect, max_loc))

        if self.angles_distance(angle0, angle1) > THRESH_ANGLE_DISTANCE:
            return prev_angle
        elif prev_angle is None:
            return angle1
        else:
            return self.clip_angle(prev_angle + self.angles_difference(angle1, prev_angle) * GAIN)





        