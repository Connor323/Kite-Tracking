import cv2
import numpy as np
import time
import threading
import keras.backend as K

from config import *
from utils import *

SPLIT_SIZE = 51

def preprocess(image):
    image = image / 255
    image -= image.mean()
    return image

def CNN_Detection(image):
    """
    Use MLP for localization for the initial detection or general localization. 

    Params:
        image: current frame
    Return:
        bbox
    """

    h, w = image.shape[:2]

    def sliding_window_mp():
        blocks, points = [], []

        num_block_x = (w - SPLIT_SIZE) // STEP_SIZE[0] + 1
        num_block_y = (h - SPLIT_SIZE) // STEP_SIZE[1] + 1
        num_blocks = num_block_x * num_block_y

        for i in range(num_blocks):
            x = i % num_block_x * STEP_SIZE[0]
            y = i // num_block_x * STEP_SIZE[1]
            patch = image[y:y + SPLIT_SIZE, x:x + SPLIT_SIZE]
            redV = redFilter(patch)
            if redV < 0.1: 
                continue
            blocks.append(preprocess(patch))
            points.append([x, y])
        return np.array(blocks), np.array(points)

    def work_bg(results):
        im_windows, points = sliding_window_mp()
        if not len(im_windows): 
            return 

        probs = bg_clf.predict(im_windows)
        
        for (x, y), prob in zip(points, probs): 
            pred = np.argmax(prob)
            if pred == 0: 
                currScore = float(prob[pred])
                tmp = (x, y, int(SPLIT_SIZE), int(SPLIT_SIZE), currScore)
                results.append(tmp)

    def redFilter(patch):
        patch = patch.astype(int)
        tmp = patch[..., 2] / (patch[..., 0] + patch[..., 1])
        num = np.sum(tmp > 0.9)
        return num / np.prod(patch.shape[:2])

    # Assign jobs
    tic = time.time()
    results = []
    work_bg(results)

    if not len(results): 
        return None

    if DEBUG_MODE:
        print("Total time: %.5fs" % (time.time() - tic))

    # Get final result
    detections = []
    final_select = None
    score = 0
    for detection in results:
        detection = np.array(detection)
        detections.append(detection[:4])
        if score < detection[4]:
            score = detection[4]
            final_select = detection[:4]
    if DEBUG_MODE:
        print("Final score: %f, total number of detections: %d" % (score, len(detections)))

    # If visualize is set to true, display the working
    # of the sliding window 
    if SHOW_RESULT: 
        clone = image.copy()
        for x1, y1, _, _ in detections:
            # Draw the detections at this scale
            x1, y1 = int(x1), int(y1)
            cv2.rectangle(clone, (x1, y1), (x1 + SPLIT_SIZE, y1 +
                SPLIT_SIZE), (0, 0, 0), thickness=2)

        # Draw current best
        if final_select is not None:
            x1, y1, _, _ = final_select
            x1, y1 = int(x1), int(y1)
            cv2.rectangle(clone, (x1, y1), (x1 + SPLIT_SIZE, y1 +
                SPLIT_SIZE), (0, 255, 0), thickness=2)
        clone_resize = cv2.resize(clone, VIZ_SIZE)
        MLP_RECORD[0] = clone_resize

    if score >= PROB_CRITERIA:
        return tuple(final_select)
    else:
        return None

def CNN_Verify(image, bbox):
    """
    Use CNN to verify if the bbox is correct
    """
    bbox = np.array(bbox).astype(int)
    patch = cropImage(image, bbox)
    if patch is None:
        return False
    input_img = np.array([preprocess(patch.astype(np.float64))])
    prob = bg_clf.predict(input_img)[0]
    pred = np.argmax(prob)
    return pred == 0
