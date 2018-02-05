This program uses [KCF](https://github.com/uoip/KCFnb) tracker and HOG + MLP for object tracking and localization.  

**Still in progress**

## Requirement
```
skimage
cv2
imageio
numba
sklearn
imutils
```

## Usage
```
python tracking.py 
```
For the setting details, please read the comments in config.py.

## Workflow 
1. Use Multilayer Perceptron (MLP) for initial bounding box detection
2. Input the intial box to the KCF tracker and start tracking
3. At the meanwhile, use background substraction (BS) to extract a rough foreground
4. Crop the binary image of BS based on the box from KCF and based on the foreground area to determine if tracking succeeds. 
5. If tracking failed, input the patches based on all foreground regions from BS to the HOG feature extractor with another MLP classifier to detect the bounding box. 
6. If tracking succeeded, go to step 3 processing next frame.
7. If tracking failed after step 5, stop the program. 

## Note
1. To detect the initial box, we still need the MLP with HOG input for localization, which takes about 2.5 sec at the beginning, since BS need a few frames to model the background.
2. Instead of using MLP for each frame to determine the success of detection, I simply use the area of foreground inside the box obtained from background subtraction, since BS can segment the kite really well (although there are still some noise on the wires and artifacts on ground). 
3. When we detected tossing target (based on 2), we use a MLP with BS result as input (I trained a 2ed MLP with binary result from BS and HOG as features) to de the localization. Moreover, instead of scanning the whole image, I only compute the region with foreground from BS result; therefore, the processing time is really fast (2-3 ms).
4. Since for training the 2ed MLP I only use the binary image rather than RBG image (I tried with RGB and it's only really good), our algorithm can deal with all kinds on kite with different color (the shape needs to be close to the red-white one). Note that we still need to train the 1st MLP for the initial box detection if we change the kite or don't want to mark the first box. 
5. As shown in the video, although BS needs a lot computation, our overall speed still can achieve 30-40 fps. 

## TODO:
 - [x] I did't normalize the histogram; therefore, it's sensitive with the difference of light condition during the interruption.
 - [x] SIFT is rotation invariant, but it's scaling invariant in a limited range; therefore, in the current implementation, I set a fixed search area when switching to SIFT.
 - [x] Test with more data. 
 - [x] Speed up localization process.
 - [x] Improve the accuracy of MLP classifier. 
 - [x] Improve the performance (bbox is not in the middel) 
