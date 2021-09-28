from typing import ClassVar
import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('albion_cabbage.jpg', cv.IMREAD_UNCHANGED)
result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

# get the best match position
min_Val, max_Val, min_Loc, max_Loc = cv.minMaxLoc(result)

print(f'best match top left position:{str(max_Loc)}')
print(f'best match confidence:{max_Val}')

threshold = 0.8
if max_Val >= threshold:
    print('found needle.')
    # get dimensions of the needle image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    top_left = max_Loc
    bottom_right = (top_left[0]+needle_w, top_left[1]+needle_h)

    cv.rectangle(haystack_img, top_left, bottom_right,
                 (0, 255.0), 2, cv.LINE_4)
    # cv.imshow('Result', haystack_img)
    # cv.waitKey(0)
    cv.imwrite('result.jpg', haystack_img)
else:
    print('Needld not found.')
