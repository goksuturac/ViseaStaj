import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
Histogram eşikleme == Histogram thresholding:
-Simple Thresholding:
    cv.THRESH_BINARY
    cv.THRESH_BINARY_INV
    cv.THRESH_TRUNC
    cv.THRESH_TOZERO
    cv.THRESH_TOZERO_INV
-Adaptive Thresholding:
    cv.ADAPTIVE_THRESH_MEAN_C: The threshold value is the mean of the neighbourhood area minus the constant C.
    cv.ADAPTIVE_THRESH_GAUSSIAN_C: The threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C.
-OTSU's Thresholding
"""
img = cv2.imread('images//input//lowContrast.png', cv2.IMREAD_GRAYSCALE)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

original_his = cv2.calcHist([img], [0], None, [256], [0, 256])
binary_his = cv2.calcHist([thresh1], [0], None, [256], [0, 256])
binary_inv_his = cv2.calcHist([thresh2], [0], None, [256], [0, 256])


titles = ['Original Image','BINARY','BINARY_INV']
images = [img, thresh1, thresh2]
for i in range(3):
    plt.subplot(1,3, i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.savefig("images//output//threshed_img.png")
plt.show()

# equalized_image = cv2.equalizeHist(image)
# adaptive_threshold = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imshow("Original Image", image)
# cv2.imshow("Equalized Image", equalized_image)
# cv2.imshow("Adaptive Thresholding Result", adaptive_threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

