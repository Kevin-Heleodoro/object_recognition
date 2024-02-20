## Task 1:

Options for thresholding values:

-   Find a value in the valley between two peaks of an image's histogram.
-   Pick a mid-range value of 128
-   Calculate the mean/median pixel value of the image.
-   Otsu's method - minimizes intra-class variance
-   Adaptive thresholding - calculates locally for different regions of the image.

https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
https://docs.opencv.org/4.x/db/d8e/tutorial_threshold.html
https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html

> Compare the result if applying a gaussian blur to the image. How much of a benefit does it provide?

## Task 2:

Clean up the binary image:

-   Use morphological filtering to clean up the images.
