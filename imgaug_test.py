import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import random

img=cv2.imread('./images/20210427_113656.png')

step=iaa.Affine(shear=(-20,20))

seq=iaa.Sequential(step)

img_aug=seq(image=img)

cv2.imshow("img transf",img_aug)

cv2.waitKey(0)