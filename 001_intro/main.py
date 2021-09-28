from typing import ClassVar
import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('albion_cabbage.jpg', cv.IMREAD_UNCHANGED)
result = cv.matchTemplate(haystack_img, needle_img, cv.TM_SQDIFF_NORMED)
print(result)
threshold = 0.17
locations = np.where(result <= threshold)
# locations = list(zip(*locations[::-1]))
if locations:
    print('found needle.')
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    line_color = (0, 255, 0)
    line_type = cv.LINE_4
    for loc in zip(*locations[::-1]):
        print(loc)
        top_left = loc
        bottom_right = (top_left[0]+needle_w, top_left[1]+needle_h)
        cv.rectangle(haystack_img, top_left,
                     bottom_right, line_color, line_type)
    cv.imshow('Matches', haystack_img)
    cv.waitKey(0)
else:
    print('Needle not found.')
