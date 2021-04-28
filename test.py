import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import json

IMAGE=70


f=open('output.json','r')
data_from_labstudio=json.load(f)
f.close()

path_to_img='./dataset/'+data_from_labstudio["images"][IMAGE]["file_name"]
img=cv2.imread(path_to_img)

xy=[]
points=[]
for p in data_from_labstudio["annotations"][IMAGE]["segmentation"][0]:
    xy.append(int(p))

    if len(xy)==2:
        points.append(xy)
        xy=[]

pts=np.array(points, np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))


cv2.imshow("img with polygon",img)

cv2.waitKey(0)

print(data_from_labstudio["annotations"][IMAGE]["segmentation"])