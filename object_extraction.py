import cv2
from PIL import Image
import numpy as np


def remove_transparency(im, bg_colour=(255, 255, 255)):
    """Remove alpha channel from RGBA image"""
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg.convert('RGB')
    else:
        # return image if no alpha channel
        return im


def get_contours(layer_ref, blur_coef, binary_thresh):
    """Get object contours"""
    im = np.uint8(layer_ref * 255)
    im = Image.fromarray(im)
    rgb_map = remove_transparency(im, bg_colour=(255, 255, 255))
    rgb_map = np.array(rgb_map)
    grayM = cv2.cvtColor(rgb_map, cv2.COLOR_GRAY2RGB)
    grayM = cv2.cvtColor(grayM, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grayM, blur_coef, 0)
    ret, binary_map = cv2.threshold(blur, binary_thresh, 255, 0)
    # Get contours / clusters defined from thresholding
    contours = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours, binary_map


def get_bounding_box_properties(contours, rotated=False):
    """Return the bounding rectangle of contours."""
    if rotated:
        box_coords = cv2.minAreaRect(contours)
    else:
        x1, y1, w, l = cv2.boundingRect(contours)
        xc = x1 + (w / 2)
        yc = y1 + (l / 2)
        box_coords = ((xc, yc), (w, l), -0.0)
    box = cv2.boxPoints(tuple(box_coords))
    box = np.int0(np.around(box))
    return box, box_coords
