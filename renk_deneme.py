import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images//input//calvinHobbes.jpeg", cv2.IMREAD_COLOR)
print(image.shape)

#RGB --> gray scale
gray_scale_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# cv2.imshow("Gray Image", gray_scale_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("images//output//gray_image.jpg", gray_scale_img)


#Gray-scale --> black&white
#threshold değeri kullanılarak piksellerdeki renk tonu 2 ye indirildi
_, black_white_image = cv2.threshold(gray_scale_img, 127, 255, cv2.THRESH_BINARY)

# cv2.imshow("Black and White Image", black_white_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("images//output//black_white_image.jpg", black_white_image)


#RGB --> HSV
hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
# cv2.imshow("HSV", hsv_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("images//output//hsv.jpg", hsv_img)




fig, axes = plt.subplots(2, 2, figsize=(6, 8))  

axes[0,0].imshow(image)
axes[0,0].set_title('Normal Image')

axes[0,1].imshow(gray_scale_img, cmap='gray')
axes[0,1].set_title('Turned Gray-scale Image')

axes[1,0].imshow(black_white_image, cmap='gray')
axes[1,0].set_title('Turned Black&White Image')

axes[1,1].imshow(hsv_img)
axes[1,1].set_title('Turned HSV Image')

plt.show()
fig.savefig("images//output//changed_color.jpg")
