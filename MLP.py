from skimage.feature import hog
import cv2
import numpy as np
import time
import threading

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(image.shape[0]/2, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def swapChannels(image):
    image = image.copy()
    tmp = image[..., 0].copy()
    image[..., 0] = image[..., 2].copy()
    image[..., 2] = tmp.copy()
    return image

def MLP_Detection_MP(image, clf, total_num_thread=8, criteria=0.9999, min_wdw_sz=(51, 51), step_size=(25, 25), visualize_det=True, record_size=(512, 512)):
    def sliding_window_mp(thread_id):
        blocks = []
        for y in xrange(image.shape[0]/2 + thread_id * step_size[1], image.shape[0], step_size[1] * total_num_thread):
            for x in xrange(thread_id * step_size[0], image.shape[1], step_size[0] * total_num_thread):
                blocks.append((x, y, image[y:y + min_wdw_sz[1], x:x + min_wdw_sz[0]]))
        return blocks

    def work(thread_id, total_num_thread, image, result):
        blocks = sliding_window_mp(thread_id)
        for idx, (x, y, im_window) in enumerate(blocks):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue

            # Calculate the HOG features
            fd = [hog(im_window[..., i], orientations=9, 
                                         pixels_per_cell=(8, 8), 
                                         cells_per_block=(3, 3), 
                                         block_norm="L2", 
                                         visualise=False) for i in range(3)]
            fd = np.array(fd)
            fd = [fd.reshape(fd.size)]
            pred = clf.predict(fd)
            if pred == 1:
                currScore = float(clf.predict_proba(fd)[0][pred])
                tmp = (x, y, int(min_wdw_sz[0]), int(min_wdw_sz[1]), currScore)
                result.append(tmp)
            # print "Thread: %d, Number of result: %d / %d" % (thread_id, idx, len(blocks))
        
    # Swap image channel from BGR to RGB
    image = swapChannels(image)
    h, w = image.shape[:2]

    # Assign jobs
    threads = []
    results = [[] for _ in range(total_num_thread)]
    for thread_id, result in enumerate(results):
        t = threading.Thread(target=work, args=(thread_id, total_num_thread, image, result))
        t.start()
        threads.append(t)

    # Wait for computing
    tic = time.time()
    still_alive = True
    while still_alive:
        still_alive = False
        for t in threads:
            if t.isAlive():
                still_alive = True
    print "Total time: %.2fs" % (time.time() - tic)

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
    print "Final score: ", score
    # If visualize is set to true, display the working
    # of the sliding window 
    if visualize_det: 
        clone = image.copy()
        for x1, y1, _, _ in detections:
            # Draw the detections at this scale
            x1, y1 = int(x1), int(y1)
            cv2.rectangle(clone, (x1, y1), (x1 + min_wdw_sz[1], y1 +
                min_wdw_sz[0]), (0, 0, 0), thickness=2)

        # Draw current best
        if final_select is not None:
            x1, y1, _, _ = final_select
            x1, y1 = int(x1), int(y1)
            cv2.rectangle(clone, (x1, y1), (x1 + min_wdw_sz[1], y1 +
                min_wdw_sz[0]), (0, 255, 0), thickness=2)
        clone_resize = cv2.resize(clone, record_size)
        clone_resize = swapChannels(clone_resize)
        cv2.imshow("Sliding Window in Progress", clone_resize)

    if score >= criteria:
        return tuple(final_select)
    else:
        return None


def MLP_Detection(image, clf, criteria=0.9999, min_wdw_sz=(51, 51), step_size=(25, 25), visualize_det=True, record_size=(512, 512)):
    # Swap image channel from BGR to RGB
    image = swapChannels(image)
    # List to store the detections
    detections = []
    final_select = None
    score = 0
    # This list contains detections at the current scale
    cd = []
    for (x, y, im_window) in sliding_window(image, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        # Calculate the HOG features
        fd = [hog(im_window[..., i], orientations=9, 
                                     pixels_per_cell=(8, 8), 
                                     cells_per_block=(3, 3), 
                                     block_norm="L2", 
                                     visualise=False) for i in range(3)]
        fd = np.array(fd)
        fd = [fd.reshape(fd.size)]
        pred = clf.predict(fd)
        if pred == 1:
            # currScore = clf.decision_function(fd)
            currScore = clf.predict_proba(fd)[0][pred]
            print  "Detection:: Location -> ({}, {})".format(x, y)
            print "    Confidence Score {}".format(currScore)
            detections.append((x, y, int(min_wdw_sz[0]), int(min_wdw_sz[1])))
            cd.append(detections[-1])
            # Select the one with max score
            if score < currScore:
                print "Current best! \n"
                final_select = np.array(detections[-1]).tolist()
                score = currScore

        # If visualize is set to true, display the working
        # of the sliding window
        if visualize_det:
            clone = image.copy()
            for x1, y1, _, _ in cd:
                # Draw the detections at this scale
                cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                    im_window.shape[0]), (0, 0, 0), thickness=2)
            cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                im_window.shape[0]), (255, 255, 255), thickness=2)
            
            # Draw current best
            if final_select is not None:
                x1, y1, _, _ = final_select
                cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                    im_window.shape[0]), (0, 255, 0), thickness=2)
            clone_resize = cv2.resize(clone, record_size)
            clone_resize = swapChannels(clone_resize)
            cv2.imshow("Sliding Window in Progress", clone_resize)
            cv2.waitKey(30)
    if score >= criteria:
        return tuple(final_select)
    else:
        return None