This program uses MIL tracker and SIFT feature descriptor for object tracking and localization. 

The general method is: we first use MIL with initial box tracking the object and when it fails (this part is little tricky to implement, since MIL fails to recognize when it losses the target. Therefore, I also compute the histogram for each small patch surrounding by the current box and use the l2 distance as a criteria to determine if tracking succeed), we use SIFT to do the object localization using the kite templates I cropped from some images. If SIFT succeeds, we continue using MIL tracking, otherwise, we stop the process. The attached result is from the image Dianjing sent me yesterday and it seems working well when interruption happens (By increasing the number of templates, we can increase success rate of SIFT).

## TODO:
 - [x] I did't normalize the histogram; therefore, it's sensitive with the difference of light condition during the interruption.
 - [ ] SIFT is rotation invariant, but it's scaling invariant in a limited range; therefore, in the current implementation, I set a fixed search area when switching to SIFT.
 - [ ] Test with more data. 