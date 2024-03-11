import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("images//input//lowContrast.png")
print(image.shape)

cropped_img = image[25:125, 50:150]

fig, axes = plt.subplots(1, 2, figsize=(4, 4))  

axes[0].imshow(image)
axes[0].set_title('Normal Image')

axes[1].imshow(cropped_img)
axes[1].set_title('Cropped Image')
plt.show()
plt.savefig("images//output//cropped_img.jpg")