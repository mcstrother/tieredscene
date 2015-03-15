# Basic Info #

Title: Reimplementation of Tiered Scene Labeling with Dynamic Programming
Team members: Marshall Strother

## Introduction ##

For this project, I reimplemented the algorithm described by Felzenszwalb and Veksler for scene segmentation.  The algorithm guarantees a time complexity O(mn<sup>2</sup>k<sup>2</sup>), where m is the width of the image, n is the height, and k is the number of regions into which the image is to be segmented, as long as the segmentation desired qualifies as "tiered."  A "tiered" segmentation in this context is constructed as follows:
  * The pixels are divided into 3 regions, "top", "middle", and "bottom".
  * Pixels in the "top" region are assigned the label "top".  The pixels in the "bottom" region are assigned the label "bottom".
  * Pixels in the middle region are assigned labels from a finite set of any number of elements.
  * All pixels in the middle region of a single column of the image must be assigned the same label.

Thus a tiered segmentation is the same as assigning each column of the image a triple (i,j,l), where i is the row of the first pixel in the middle region (counting down from the top), j is the row of the first pixel in the bottom region, and l is the label assigned to the middle region.  (I refer to this triple below as the column's "state".)

## References ##

[Original paper by Felzenszwalb and Veksler](http://people.cs.uchicago.edu/~pff/papers/dpseg.pdf)

## Technical Description ##

For a full detailed description, see the original paper linked above.  A brief description of key elements of the algorithm follows.

Algorithm requires that it is given two loss functions which are to be minimized by the final labeling: a DataLossFunction and a SmoothnessLossFunction.  The DataLossFunction can be any function that takes a pixel location and a label and in O(1) time returns a loss value that would result from giving that pixel that label.  This loss must be independent of the labels of any other pixels in the image.  The SmoothnessLossFunction is similar, but the loss for a given pixel/label pair may depend on the labels assign to the pixels just to the left and just above the pixel.

The algorithm begins by calculating a series of integral images which allows the data loss and vertical component of the smoothness loss to be calculated in O(1) for any state/column pair.  A similar trick allows us to calculate the horizontal component of the smoothness loss give a column number, its state, and the state of the column to its left.

Then we build a dynamic programming table as follows
```
for every possible value of l for this column:
    for every possible value of l for the previous column:
        **
        for every possible value of i for this column:
            for every possible value of j for this column:
                find the i and j of the best state for the previous column if the state for this column is set to be (i, j, l)

```

Normally the time complexity of this last "find" would be O(n^2).  However a feature of the integral image trick for the horizontal smoothness component is that for a given relationship between the i's and j's of two states (e.g. i1<=j1<=i2<=j2 or i1<=i2<=j1<=j2), the horizontal smoothness loss between two states can be expressed as the sum of functions that depend only on i1, i2, j1, and j2.  This allows us to build a table at the place in the code marked by the  which allows us to do this "find" in O(1) time.

## Experimental Results ##


At the time of this writing, I was unable to finish debugging the speed-up described in the last paragraph above.  (For the sake of grading, I hope that my [mercurial changelog](http://code.google.com/p/tieredscene/source/list) and [my source code](http://code.google.com/p/tieredscene/source/browse/#hg/tieredscene) will inspire some mercy and show that my failure was not for lack of effort or advanced planning.)  Shown below are results for the buggy version of the fast algorithm and a brute-force version of the algorithm which takes advantage of the loss-function caching, and dynamic programming, but not the final speed up.  (The slower version had to be run on much smaller images in order to have results in a reasonable amount of time.)

The loss functions used for image 1 were tailored to give known results.  The loss function for image 2 is the same as the one used in the paper with no data loss.  The "per-pixel class confidences" alluded to in the paper were unavailable.

(All images can be downloaded on the "downloads" section of this project.)

### Image 1 ###
![http://tieredscene.googlecode.com/files/testimage.png](http://tieredscene.googlecode.com/files/testimage.png)

![http://tieredscene.googlecode.com/files/demo1_out_dp.png](http://tieredscene.googlecode.com/files/demo1_out_dp.png)

![http://tieredscene.googlecode.com/files/demo1_out_brute.png](http://tieredscene.googlecode.com/files/demo1_out_brute.png)

### Image 2 ###

![http://tieredscene.googlecode.com/files/Taj_Mahal_small.jpg](http://tieredscene.googlecode.com/files/Taj_Mahal_small.jpg)

![http://tieredscene.googlecode.com/files/demo2_out_dp.png](http://tieredscene.googlecode.com/files/demo2_out_dp.png)

![http://tieredscene.googlecode.com/files/demo2_out_brute.png](http://tieredscene.googlecode.com/files/demo2_out_brute.png)




## Discussion ##

### Accuracy ###

The algorithm converges to a global minimum (both in theory and in my trials).  The accuracy of the segmentation is thus entirely dependent on the definitions of the loss functions.  I did not have time to run many trials, but from those I've seen, it appears that the algorithm is rather sensitive to the balance of the data loss and horizontal loss functions.  From my results and from those in the paper, it appears that a rather significant amount of by-hand tweaking, especially of the smoothness loss function, is necessary to achieve viable results.

### Speed ###

While the authors of the paper claim that their implementation terminates in less than 2 minutes in most cases, I found that my implementation (in Python with heavy use of the numpy library) generally took around 4 minutes to terminate with an approximately 50x50 pixel image.   The majority of this time was spent in generating the tables for the final speed-up.  While this technically runs in n<sup>2</sup> time, I found it to be relatively slow compared to generation of the loss function caches, which supposedly run in O(n<sup>2</sup>**m\*k<sup>2</sup>) time.**

The implementation without the final speedup was, of course, much slower.  The largest image I could process using that implementation was about 20x15 pixels, which took about 15 minutes to terminate.

## Future Work ##

Obviously, I would complete the part of the program that currently doesn't work.
After that, I believe the most promising work suggested by the authors is the use of "shape priors" to automatically generate the loss functions in a way that targets segmenting specific foreground shapes from the rest of the scene.  An interesting extension would be to extend the work described in the paper to situations in which several of the same prior are present in the foreground of a single scene.