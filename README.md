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

"Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";

### Sunglasses

![sunglasses](/img/sunglasses_original.png)

type 1
value 70
![sunglasses_1](/img/task_1/sunglasses_1.png)

### Fork

![fork](/img/fork_original.png)

type 1
value 70
![fork_1](/img/task_1/fork_1.png)

type 1
value 152
![fork_2](/img/task_1/fork_2.png)

### Pen

![pen](/img/pen_original.png)

type 1
value 70
![pen_1](/img/task_1/pen_1.png)

## Task 2:

Clean up the binary image:

-   Use morphological filtering to clean up the images.
    -   noise reduction - morphological opening - white spots
    -   filling holes - morphological closing - black spots

I will use the filling holes approach since the thresholded images seem to be displaying a lot of black spots.
**What are the black spots?**

### Sunglasses

> Kernel size of 5 on morph_close did not produce any changes to the black spots present in sunglasses_1
> No changes when using the morph_open with kernel size 5.

> Setting a kernel size of 25 filled in the gaps and overlapped into the shadows.

type 1
value 70
kernel size 25
![sunglasses_2](img/task_2/sunglasses_2.png)

type 1
value 70
kernel size 15
![sunglasses_3](img/task_2/sunglasses_3.png)

Differences between 25 and 15 seem negligible. However, dropping down to kernel size 10 allowed for the black spots to show again.

### Fork

value 152
kernel size 15
![fork_3](img/task_2/fork_3.png)

The fork needs a lot more fine tuning. This is most likely due to the metallic aspects of it. There are many areas of light and dark in the same place.

### Pen

value 70
kernel size 15
![pen_2](img/task_2/pen_2.png)

## Segment into regions
