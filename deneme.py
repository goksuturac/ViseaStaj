import cv2
import matplotlib.pyplot as plt


image = cv2.imread("images\\lowContrast.jpeg", cv2.IMREAD_COLOR)
print(image.shape)


new_height = 200
new_width = 250

resized_image = cv2.resize(image, (new_width, new_height))
cv2.imwrite("images\\resized_image.jpg", resized_image)
print(resized_image.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))  

axes[0].imshow(image)
axes[0].set_title('Normal Image')

axes[1].imshow(resized_image)
axes[1].set_title('Resized Image')

fig.savefig("images/normal_resized.jpg")


#Mean Blur
mean_blurred = cv2.blur(image,(5,5))

#Gussian Blur
gaussian_blurred = cv2.GaussianBlur(image,(5,5),0) #burdaki 0 değeri standart sapmayı otomatik hesaplamaya yarar

#Median Blur
median_blurred = cv2.medianBlur(image,(3))



fig, axes = plt.subplots(2, 2, figsize=(10, 10))  

axes[0,0].imshow(image)
axes[0,0].set_title('Normal Image')

axes[0,1].imshow(mean_blurred)
axes[0,1].set_title('Mean Blurred Image')

axes[1,0].imshow(gaussian_blurred)
axes[1,0].set_title('Gaussian Blured Image')

axes[1,1].imshow(median_blurred)
axes[1,1].set_title('Median Blurred Image')

fig.savefig("images/blurred_images.jpg")


cv2.imshow("Blurred Image", gaussian_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()