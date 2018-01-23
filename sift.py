import cv2
import glob
import numpy as np

class SIFT:
    """docstring for SIFT"""
    def __init__(self, ROI, template_path):
        self.ROI =  ROI # preset searching ROI
        self.templates = [cv2.imread(file, 0) for file in glob.glob(template_path)] # query_image

    def compute(self, image):
        image = image[self.ROI[1]:self.ROI[1]+self.ROI[3], self.ROI[0]:self.ROI[0]+self.ROI[2]]

        for template in self.templates:
            # Initiate SIFT detector
            sift = cv2.xfeatures2d.SIFT_create()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(template,None)
            kp2, des2 = sift.detectAndCompute(image,None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1,des2,k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            if len(good) != 0:
                pt = np.array(kp2[good[0].trainIdx].pt)
                pt[0] += self.ROI[0]
                pt[1] += self.ROI[1]
                return pt

    def getBoxFromPt(self, pt, bbox):
        return (pt[0] - bbox[2]/2, pt[1] - bbox[3]/2, bbox[2], bbox[3])