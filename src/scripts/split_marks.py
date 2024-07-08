import cv2
import sys
import os



nRows = 5
# Number of columns
mCols = 10

# Reading image
img = cv2.imread("data/digits3_0.png")
#print img

#cv2.imshow('image',img)
#cv2.waitKey()

# Dimensions of the image
sizeX = img.shape[1]
sizeY = img.shape[0]

top_margin = 150
left_margin = 5
mark_width = 160
mark_height = 400
for i in range(0, nRows):
    for j in range(0, mCols):
        h1 = int(i*mark_height) + top_margin
        h2 = int(i*mark_height + mark_height) + top_margin
        w1 = int(j*mark_width) + left_margin
        w2 = int(j*mark_width + mark_width) + left_margin
        roi = img[h1:h2, w1:w2]
        # cv2.imshow('rois'+str(i)+str(j), roi)
        # cv2.imwrite('C:\sources\simple-ocr-opencv\simpleocr\data\splits\mark_'+str(i)+"-"+str(j)+".jpg", roi)

cv2.waitKey()