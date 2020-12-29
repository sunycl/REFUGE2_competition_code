
import cv2
import numpy as np


"""This program is used to get the CDR value."""

def fitEllipse(LabelImg0):



    LabelImg = LabelImg0.copy()
    if len(LabelImg.shape) == 3:
        LabelImg = LabelImg[:,:,0]


    disc_intensity = 150
    cup_intensity = 255

    LabelImg[LabelImg > disc_intensity + 50] = cup_intensity
    LabelImg[np.bitwise_and(LabelImg > 0, LabelImg < 255)] = disc_intensity

    LabelCup = np.uint8(LabelImg == 255)
    LabelDisc = np.uint8(LabelImg > 0)

    ############################################################
    ############################################################
    """Validate the Segmentation Result of Cup and disc, 
    by 1. pixel count
    """

    error_codes = [0, 0, 0, 0]  ##[disc_seg, cup_seg, disc_fit, cup_fit]

    disc_pixcnt = np.count_nonzero(LabelDisc)
    cup_pixcnt = np.count_nonzero(LabelCup)
    seg_flag = True
    pix_threshold = 1000 * LabelImg.shape[0] * LabelImg.shape[1] / (512 * 512)
    if disc_pixcnt < pix_threshold:
        seg_flag = False
        error_codes[0] = -1
        print('WARNING: Disc is not detected')
    if cup_pixcnt < disc_pixcnt * 0.01:
        seg_flag = False
        error_codes[1] = -1
        print('WARNING: Cup is not detected')
    # print('#' * 20)
    # print('pixle count:', disc_pixcnt, cup_pixcnt)

    ############################################################


    """Use direct Polar Transformation centered at optic disc center to get ISNT"""
    if seg_flag:

        CupDisc_EllipseFit = np.zeros(LabelImg.shape[:2], np.uint8)

        _, contours_Cup, hierarchy_Cup = cv2.findContours(LabelCup, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas_cup = [cv2.contourArea(c) for c in contours_Cup]
        max_index_cup = np.argmax(areas_cup)
        cup_blob = contours_Cup[max_index_cup]

        "cup_elipse format: ((center col, center row), (major axis, minor axis), rotation angle in degrees)"
        cup_elipse = cv2.fitEllipse(cup_blob)
        # cup_center = [int(round(cup_elipse[0][1])), int(round(cup_elipse[0][0]))]
        # cup_MajorAxis = int(round(cup_elipse[1][0]))
        # cup_MinorAxis = int(round(cup_elipse[1][1]))
        # cup_angle = cup_elipse[2]
        # if cup_MajorAxis < cup_MinorAxis:
        #     cup_MinorAxis, cup_MajorAxis = cup_MajorAxis, cup_MinorAxis
        #     cup_angle = 90 - cup_angle

        _, contours_Disc, hierarchy_Disc = cv2.findContours(LabelDisc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas_disc = [cv2.contourArea(c) for c in contours_Disc]
        max_index_disc = np.argmax(areas_disc)
        disc_blob = contours_Disc[max_index_disc]

        "disc_elipse format: ((center col, center row), (major axis, minor axis), rotation angle in degrees)"
        disc_elipse = cv2.fitEllipse(disc_blob)
        # disc_center = [int(round(disc_elipse[0][1])), int(round(disc_elipse[0][0]))]
        # disc_MajorAxis = int(round(disc_elipse[1][0]))
        # disc_MinorAxis = int(round(disc_elipse[1][1]))
        # disc_angle = disc_elipse[2]
        # if disc_MajorAxis < disc_MinorAxis:
        #     disc_MinorAxis, disc_MajorAxis = disc_MajorAxis, disc_MinorAxis
        #     disc_angle = 90 - disc_angle
        #
        cv2.ellipse(CupDisc_EllipseFit, disc_elipse, disc_intensity, -1)
        total_area_cnt_disc = np.count_nonzero(CupDisc_EllipseFit>0)

        cv2.ellipse(CupDisc_EllipseFit, cup_elipse, cup_intensity, -1)
        total_area_cnt_cupdisc = np.count_nonzero(CupDisc_EllipseFit > 0)
        cup_out_ratio = (total_area_cnt_cupdisc - total_area_cnt_disc) / float(cup_pixcnt)

        if cup_out_ratio < 0.2:
            pass
        else:
            seg_flag = False   ##this indicate that cup is located outside disc
            print('WARNING: Segmentation Error -- cup is located outside disc')


        """Draw the original contour on the labelimg"""
        # LabelImg_show = cv2.cvtColor(LabelImg, cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(LabelImg_show, contours_Cup, max_index_cup, (0,255,0), 3)
        # cv2.drawContours(LabelImg_show, contours_Disc, max_index_disc, (0,255,0), 3)

        # cv2.drawContours(ImgShow, contours_Cup, max_index_cup, (255, 0, 0), 3)
        # cv2.drawContours(ImgShow, contours_Disc, max_index_disc, (0, 255, 0), 3)
        # cv2.imwrite('ImgShow.png', ImgShow)

        ############################################################
        """Validate the Segmentation Result of Cup and disc, 
        2. comparing to original and ellipsefit"""


        Cup_EllipseFit = np.uint8(CupDisc_EllipseFit == 255)
        Disc_EllipseFit = np.uint8(CupDisc_EllipseFit > 0)

        cup_ellipsefit_dice = dice(Cup_EllipseFit, LabelCup)
        disc_ellipsefit_dice = dice(Disc_EllipseFit, LabelDisc)
        # print('Ellipse Fitting Dice:', cup_ellipsefit_dice, disc_ellipsefit_dice)

        if disc_ellipsefit_dice < 0.95:
            error_codes[2] = -1
            print('WANRING: the disc segmentation might be problematic. ')
        if cup_ellipsefit_dice < 0.9:
            error_codes[3] = -1
            print('WANRING: the cup segmentation might be problematic. ')

    else:

        CupDisc_EllipseFit = None
        disc_elipse = None
        cup_elipse = None

    return CupDisc_EllipseFit, disc_elipse, cup_elipse, seg_flag, error_codes




def dice(pred, targs):
    pred = np.uint8(pred > 0)
    targs = np.uint8(targs > 0)
    return 2. * np.sum(pred * targs) / (np.sum(pred) + np.sum(targs))