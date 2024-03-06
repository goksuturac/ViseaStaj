import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images//input//lowContrast.png', cv2.IMREAD_GRAYSCALE)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

original_his = cv2.calcHist([img], [0], None, [256], [0, 256])
binary_his = cv2.calcHist([thresh1], [0], None, [256], [0, 256])
binary_inv_his = cv2.calcHist([thresh2], [0], None, [256], [0, 256])


titles = ['Original Image', 'BINARY', 'BINARY_INV']
images = [img, thresh1, thresh2]
histograms = [original_his, binary_his, binary_inv_his]

plt.figure(figsize=(5, 10))

for i in range(3):
    plt.subplot(3, 2, i*2+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

    plt.subplot(3, 2, i*2+2)
    plt.plot(histograms[i], color='black')
    plt.title('Histogram')
    plt.grid(True)

plt.tight_layout()
plt.savefig("images//output//threshed_img_his.png")
plt.show()
