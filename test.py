import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import json

IMAGE=0


f=open('result.json','r')
data_from_labstudio=json.load(f)
f.close()
path_to_img=data_from_labstudio["images"][IMAGE]["file_name"]
#path_to_img='./dataset/'+data_from_labstudio["images"][IMAGE]["file_name"]
img=cv2.imread(path_to_img)

#DRAW SEGMENTATION

xy=[]
points=[]
for p in data_from_labstudio["annotations"][IMAGE]["segmentation"][0]:
    xy.append(int(p))

    if len(xy)==2:
        points.append(xy)
        xy=[]

pts=np.array(points, np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255)) #'''


#DRAW BBOX
pts=data_from_labstudio["annotations"][IMAGE]["bbox"]
xy1=(int(pts[0]),int(pts[1]))
# x+w, y+h
xy2=(int(pts[0]+pts[2]),int(pts[1]+pts[3]))
color=(0,255,0)
cv2.rectangle(img,xy1,xy2,color,3)
#'''
'''
cv2.imshow("img with polygon",img)
cv2.waitKey(0)'''
cv2.imwrite("test_img.jpg",img)

print(data_from_labstudio["annotations"][IMAGE]["segmentation"])