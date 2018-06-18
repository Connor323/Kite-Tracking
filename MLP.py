from skimage.feature import hog
import cv2
import numpy as np
import time
import threading

from config import *
from utils import *

def MLP_Detection_MP(image, bs_image, centroids):
    """
    Use MLP for localization for the initial detection or general localization. 

    Params:
        image: current frame
        bs_image: binary image 
        centroids: centroids from the binary image
    Return:
        bbox
    """

    h, w = image.shape[:2]

    def sliding_window_mp(thread_id):
        blocks = []

        num_block_x = (w - BBOX_SIZE[0]) // STEP_SIZE[0] + 1
        num_block_y = (h // 2 - BBOX_SIZE[1]) // STEP_SIZE[1] + 1
        num_blocks = num_block_x * num_block_y

        for i in xrange(thread_id, num_blocks, NUM_THREADS_TRACKING):
            x = i % num_block_x * STEP_SIZE[0]
            y = i // num_block_x * STEP_SIZE[1]
            blocks.append((x, y, image[y:y + BBOX_SIZE[1], x:x + BBOX_SIZE[0]]))
        return blocks
    
    def preprocess(image):
        image = image / 255
        image -= image.mean()
        return image

    def work_bg(image, centroid, result):
        x, y = centroid[0] - BBOX_SIZE[0] // 2, centroid[1] - BBOX_SIZE[1] // 2
        im_window = image[y: y + BBOX_SIZE[1], 
                          x: x + BBOX_SIZE[0]]
        
        if im_window.shape[0] != BBOX_SIZE[1] or im_window.shape[1] != BBOX_SIZE[0]:
            return 

        input_img = np.array([preprocess(im_window.astype(np.float64))])
        prob = bg_clf.predict(input_img)[0]
        pred = np.argmax(prob)
        if pred == 0: 
            currScore = float(prob[pred])
            tmp = (x, y, int(BBOX_SIZE[0]), int(BBOX_SIZE[1]), currScore)
            result.append(tmp)

    # Assign jobs
    tic = time.time()
    threads = []
    results = [[] for _ in range(NUM_THREADS_TRACKING)]
    for centroid, result in zip(centroids, results):
        work_bg(image, centroid, result)

    # Wait for computing
    still_alive = True
    while still_alive:
        still_alive = False
        for t in threads:
            if t.isAlive():
                still_alive = True
    if DEBUG_MODE:
        print("Total time: %.5fs" % (time.time() - tic))

    # Get final result
    detections = []
    final_select = None
    score = 0
    for result in results:
        for detection in result:
            detection = np.array(detection)
            detections.append(detection[:4])
            if score < detection[4]:
                score = detection[4]
                final_select = detection[:4]
    bs_patch = None
    if final_select is not None:
        bs_patch = cropImage(bs_image, final_select[:4])
    if DEBUG_MODE:
        print("Final score: %f, total number of detections: %d" % (score, len(detections)))

    # If visualize is set to true, display the working
    # of the sliding window 
    if SHOW_RESULT: 
        clone = image.copy()
        for x1, y1, _, _ in detections:
            # Draw the detections at this scale
            x1, y1 = int(x1), int(y1)
            cv2.rectangle(clone, (x1, y1), (x1 + BBOX_SIZE[1], y1 +
                BBOX_SIZE[0]), (0, 0, 0), thickness=2)

        # Draw current best
        if final_select is not None:
            x1, y1, _, _ = final_select
            x1, y1 = int(x1), int(y1)
            cv2.rectangle(clone, (x1, y1), (x1 + BBOX_SIZE[1], y1 +
                BBOX_SIZE[0]), (0, 255, 0), thickness=2)
        clone_resize = cv2.resize(clone, VIZ_SIZE)
        MLP_RECORD[0] = clone_resize

    if score >= PROB_CRITERIA:
        return tuple(final_select), bs_patch
    else:
        return None, None