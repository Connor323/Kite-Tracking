import cv2
import numpy as np

from config import *

class BS:
    """docstring for BS"""
    def __init__(self):
        self.frame = None
        self.bbox = None
        self.ret = False
        self.patch = None
        self.binary_image = None
        self.centroids = None
        self.new_frame = True
        self.prev_fmask = None

    def process_bs(self, low_area=10, up_area=1000, return_centroids=False):
        """
        This function applies the BS model given the current frame and implements the morphological 
        operation and region growing as the post process. 

        Params: 
            low_area: minimum area of target
            up_area: maximum area of target
            return_centroids: if True, reture the centroid of selected areas
        Return: 
            final_labels: binary image obtained from BS
            centroids: centroid of selected areas on if return_centroids == True
        """
        # process background substraction
        h, w = self.frame.shape[:2]
        
        # Downsample the image for faster speed
        image_resize = cv2.resize(self.frame, (int(w // BS_DOWNSAMPLE), int(h // BS_DOWNSAMPLE)))
        ROI_Y = int(len(image_resize) * HEIGHT_ROI_RATIO)
        image_resize = image_resize[ROI_Y:]

        # Apply BS 
        if self.new_frame or self.prev_fmask is None: 
            if BG_MODEL[0] is not None:
                fgmask = fgbg.apply(image_resize, BG_MODEL[0])
            else:
                fgmask = fgbg.apply(image_resize)
            self.prev_fmask = fgmask
        else:
            fgmask = self.prev_fmask

        if SHOW_RESULT:
            tmp_show = cv2.resize(fgmask, VIZ_SIZE, cv2.INTER_NEAREST)
            BS_ORIGIN_RECORD[0] = cv2.cvtColor(tmp_show, cv2.COLOR_GRAY2RGB)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, fgbg_kernel_open) # remove small items 
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, fgbg_kernel_close) # fill holes
        
        # obtain the regions in range of area (low_area, up_area)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask.astype(np.uint8), connectivity=4)
        select_labels = np.where((stats[..., -1] > low_area) * \
                                 (stats[..., -1] < up_area))[0]

        # refine the labels
        new_h, new_w = labels.shape[:2]
        tmp_labels = np.zeros([new_h+ROI_Y, new_w], np.uint8)
        tmp = np.zeros_like(labels).astype(np.uint8)
        for select_label in select_labels: 
            tmp[labels == select_label] = 255
        tmp_labels[ROI_Y:] = tmp
        final_labels = cv2.resize(tmp_labels, (w, h), cv2.INTER_NEAREST)
        FG_MODEL[0] = tmp
        self.binary_image = final_labels

        if SHOW_RESULT:
            tmp_show = cv2.resize(tmp_labels, VIZ_SIZE, cv2.INTER_NEAREST)
            BS_POST_RECORD[0] = cv2.cvtColor(tmp_show, cv2.COLOR_GRAY2RGB)
        
        if not return_centroids:
            return final_labels
        else:
            if len(centroids):
                centroids = centroids[select_labels]
                centroids[:, 1] += ROI_Y
                centroids *= BS_DOWNSAMPLE
                self.centroids = centroids.astype(int)
            return final_labels, centroids.astype(int)

    def centerBoxAndCrop(self, image, centroids, bbox):
        """
        Center bbox to the centroid of target, if the difference is over the threshold value.

        Params:
            image: BS result (binary image)
            centroids: a list of points
            bbox: bounding box
        Return:
            patch
            if over the threshold value
        """
        h, w = image.shape[:2]
        nd_bbox = np.array(bbox)
        c_bbox = np.array(nd_bbox[:2] + nd_bbox[2:] // 2)
        
        dists = np.linalg.norm(centroids - c_bbox, axis=1)
        min_idx = np.argmin(dists)
        min_val = dists[min_idx]

        new_bbox = nd_bbox
        new_bbox[:2] = centroids[min_idx] - nd_bbox[2:] // 2
        new_bbox = new_bbox.astype(int).tolist()
        self.bbox = new_bbox

        if min_val > RECENTER_THRESH: 
            return self.cropImage(image, new_bbox), True
        else:
            return self.cropImage(image, new_bbox), False

    def cropImage(self, image, bbox):
        """
        Crop image.

        Params:
            image: BS result (binary image)
            bbox: bounding box
        Return:
            patch
        """
        if image is None:
            return None

        h, w = image.shape[:2]

        crop_x_min = int(max(0, bbox[0]))
        crop_x_max = int(min(w - 1, bbox[0] + bbox[2]))
        crop_y_min = int(max(0, bbox[1]))
        crop_y_max = int(min(h - 1, bbox[1] + bbox[3]))
        patch = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        if patch.shape[0] != bbox[3] or patch.shape[1] != bbox[2]: # image edge case
            return None
        else:
            return patch

    def cropImageFromBS(self):
        """
        Crop patch and analysis using histogram

        Params: 
            None
        Return:
            patch of image
            if over the threshold value
        """
        image, centroids = self.process_bs(low_area=MIN_AREA, up_area=MAX_AREA, return_centroids=True)
        if len(centroids) > 0:
            patch, ret = self.centerBoxAndCrop(image, centroids, self.bbox)
        else:
            return None, True

        return patch, ret

    def updateBG(self):
        """
        Update background model for BS

        Params: 
            None
        return:
            None
        """
        if BG_MODEL[0] is None: 
            BG_MODEL[0] = fgbg.getBackgroundImage()
        else:     
            if FG_MODEL[0] is not None:
                newBG = fgbg.getBackgroundImage()
                BG_MODEL[0][FG_MODEL[0] == 0] = newBG[FG_MODEL[0] == 0]

    def cropImageAndAnalysis(self):
        """
        Determine if the current patch contains target

        Params: 
            None
        return:
            None
        """
        if UPDATE_BACKGROUND: 
            self.updateBG()

        if self.frame is None or self.bbox is None:
            self.ret = False
            self.patch = None
            return

        patch, ret = self.cropImageFromBS()
        if patch is None or ret: # crop image size is incorrect (near the edge)
            self.ret = False
            self.patch = patch
            return
        if np.sum(patch != 0) > TRACKING_CRITERIA_AREA:
            self.ret = True
            self.patch = patch
            return
        self.ret = False
        self.patch = patch
        self.new_frame = False

    def run(self):
        """
        Thread handler
        """
        while True:
            if self.new_frame: 
                self.cropImageAndAnalysis()
            if KILL_BS[0]:
                break

    def set_info(self, image, bbox):
        """
        Set current info
        """
        self.frame = image
        self.bbox = bbox
        self.new_frame = True

    def set_frame(self, image):
        self.frame = image
        self.new_frame = True

    def get_info(self):
        """
        Get current info
        """
        return self.ret, self.patch, self.bbox

    def get_binary_result(self):
        """
        Get the current binary full-size image
        """
        return self.binary_image

    def get_centroids(self):
        """
        Get the current centroids from the bs image
        """
        return self.centroids
