from skimage import io
import cv2 as cv


def imread(filename):
    origin = io.imread(filename, plugin='pil')
    img_x2 = cv.resize()
