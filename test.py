
import cv2
import imutils


image = cv2.imread('med.jpg')

image = imutils.resize(image, width=min(550, image.shape[1]))

cv2.imshow('image',image)
k = cv2.waitKey(9999999) & 0xff
