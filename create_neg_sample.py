import cv2 
import glob 

IMAGE_PATH = "/Users/hanxiang/Downloads/20180629/*.bmp"
fnames = glob.glob(IMAGE_PATH)

SPLIT_SIZE = 51
counter = 0
for fname in fnames: 
    tmp = cv2.imread(fname)
    h, w = tmp.shape[:2]
    tmp = cv2.resize(tmp, (w/2, h/2))
    
    num_block_x = (w/2 - SPLIT_SIZE) // SPLIT_SIZE + 1
    num_block_y = (h/8 - SPLIT_SIZE) // SPLIT_SIZE + 1
    num_blocks = num_block_x * num_block_y

    offset = h*3/8

    for i in range(num_blocks):
        x = i % num_block_x * SPLIT_SIZE
        y = i // num_block_x * SPLIT_SIZE
        patch = tmp[offset+y:offset+y + SPLIT_SIZE, x:x + SPLIT_SIZE]
        cv2.imwrite("samples/negs/image_%07d.png" % counter, patch)
        counter += 1