import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread("images//input//calvinHobbes.jpeg", cv2.IMREAD_GRAYSCALE)

# Histogramı hesapla
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Histogram grafiğini çiz
plt.figure(figsize=(8, 5))
plt.plot(histogram, color='black')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.grid(True)
plt.show()
