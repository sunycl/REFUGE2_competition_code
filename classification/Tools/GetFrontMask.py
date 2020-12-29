
import cv2
import numpy as np

def getFrontMask(Image, threshold=10):
    ##This program try to creat the mask for the filed-of-view
    ##Input original image (RGB or green channel), threshold (user set parameter, default 10)
    ##Output: the filed-of-view mask

    if len(Image.shape) == 3:  ##RGB image
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        Mask0 = gray >= threshold

    else:  # for green channel image
        Mask0 = Image >= threshold

    # ######get the largest blob, this takes 0.18s
    cvVersion = int(cv2.__version__.split('.')[0])

    Mask0 = np.uint8(Mask0)
    if cvVersion == 2:
        contours, hierarchy = cv2.findContours(Mask0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(Mask0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    Mask = np.zeros(Image.shape[:2], dtype=np.uint8)
    cv2.drawContours(Mask, contours, max_index, 1, -1)

    # ResultImg = Image.copy()
    # if len(Image.shape) == 3:
    #     ResultImg[Mask == 0] = (255, 255, 255)
    # else:
    #     ResultImg[Mask == 0] = 255

    return Mask
