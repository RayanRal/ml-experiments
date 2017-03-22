# import the necessary packages
import cv2

# load the image and show it
image = cv2.imread("jurassic-park-tour-jeep.jpg")
cv2.imshow("original", image)
cv2.waitKey(0)

#### RESIZE IMAGE
newWidth = 100.0
oldWidth = image.shape[1]
# ratio between new width and old width
ratio = newWidth / oldWidth
newHeight = image.shape[0] * ratio
dim = (int(newWidth), int(newHeight))

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


#### ROTATE IMAGE
# grab the dimensions of the image and calculate the center of the image
# 3rd value in shape is number of channels
(h, w) = image.shape[:2]
center = (w / 2, h / 2)

# rotate the image by 180 degrees
M = cv2.getRotationMatrix2D(center, angle=180, scale=1.0)
rotated = cv2.warpAffine(image, M, dsize=(w, h))


#### CROP IMAGE
# just take a slice of array
cropped = image[70:170, 440:540]


#### SAVE IMAGE
cv2.imwrite("thumbnail.png", cropped)

