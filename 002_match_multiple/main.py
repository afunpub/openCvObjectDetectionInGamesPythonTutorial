import cv2 as cv
import numpy as np
import os


# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Can use IMREAD flags to do different pre-processing of image files,
# like making them grayscale or reducing the size.
# https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
haystack_img = cv.imread('albion_copper.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('albion_copper_needle.jpg', cv.IMREAD_UNCHANGED)

# There are 6 comparison methods to choose from:
# TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
# You can see the differences at a glance here:
# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
# Note that the values are inverted for TM_SQDIFF and TM_SQDIFF_NORMED
result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

threshold = 0.80
locations = np.where(result >= threshold)
locations=list(zip(*locations[::-1]))
print(locations)
