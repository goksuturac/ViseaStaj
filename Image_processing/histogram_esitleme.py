import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images//input//calvinHobbes.jpeg", cv2.IMREAD_GRAYSCALE)

equalized_img = cv2.equalizeHist(image)

fig, axes = plt.subplots(1,2, figsize=(6,5))
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Orijinal Fotoğraf")

axes[1].imshow(equalized_img, cmap="gray")
axes[1].set_title("Histogramı Eşitlenmiş Fotoğraf")

plt.show()
plt.savefig("images//output//gray_histogram_esit.jpeg")

# plt.figure(figsize=(8, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(equalized_img, cmap='gray')
# plt.title('Equalized Image')
# plt.axis('off')

# plt.show()