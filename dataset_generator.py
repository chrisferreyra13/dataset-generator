import random
import json
from imgaug.augmenters.geometric import ShearX
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
import numpy as np

DATAPOINTS_PER_IMAGE=20
RESIZE=True
JSON_FILE='result.json'
OUTPUT_DIR='./dataset'

final_resize_step=iaa.Resize({"width": 640, "height": "keep-aspect-ratio"})


# Steps to apply
def gen_steps():

    STEPS=[
        iaa.Rotate((-50,-20)),
        iaa.Rotate((20,50)),
        iaa.Affine(shear=(-30,-10)),
        iaa.Affine(shear=(10,30)),
        iaa.ShearX(((-10, 10))),
        iaa.ShearY(((-10, 10))),
        iaa.Affine(scale=(0.5, 1.5)),
        iaa.Affine(scale=(1, 2)),
        iaa.Affine(translate_px={"x": (1, random.randint(300,800)), "y": (1, random.randint(300,800))}),
        [
            iaa.Affine(translate_px={"x": (1, random.randint(300,800)), "y": (1, random.randint(300,800))}),
            iaa.Affine(shear=(-15,15)),
        ],
        [
            iaa.Affine(translate_px={"x": (1, random.randint(300,800)), "y": (1, random.randint(300,800))}),
            iaa.Affine(shear=(-15,15)),
        ],
    ] 

    return STEPS


# read json
os.makedirs(OUTPUT_DIR,exist_ok=True)

f=open(JSON_FILE,'r')
data_from_labstudio=json.load(f)
f.close()


data={}
data["images"]=[]
data["annotations"]=[]

with open('output.json','w') as output_json:
    num_images=len(data_from_labstudio["images"])
    count=0

    for i in range(num_images):
        segment=data_from_labstudio["annotations"][i]["segmentation"][0]
        pts=data_from_labstudio["annotations"][i]["bbox"]
        #x,y, x+w, y+h
        bb=[pts[0], pts[1], pts[0] + pts[2], pts[1]+ pts[3]]
        poly = [ia.Polygon([
            (segment[0],segment[1]), 
            (segment[2],segment[3]), 
            (segment[4],segment[5]), 
            (segment[6],segment[7])]
            )]
        bbox=[ia.BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3])]
        
        img=cv2.imread(data_from_labstudio["images"][i]["file_name"])

        if RESIZE:
            seq=iaa.Sequential([final_resize_step])
            img_aug, polygon_aug, bbox_aug = seq(image=img, polygons=poly, bounding_boxes=bbox)
        else:
            img_aug=img
            polygon_aug=poly
            bbox_aug=bbox


        cv2.imwrite(os.path.join(OUTPUT_DIR,f"img_{count}.jpg"),img_aug)

        data["images"].append({
                "width": img_aug.shape[1],
                "height": img_aug.shape[0],
                "id": count,
                "file_name": f"img_{count}.jpg"
            })

        data["annotations"].append({
                "id": count,
                "image_id": count,
                "category_id": data_from_labstudio["annotations"][i]["category_id"],
                "bbox":[np.float64(bbox_aug[0].x1),np.float64(bbox_aug[0].y1),np.float64(bbox_aug[0].x2),np.float64(bbox_aug[0].y2)],
                "segmentation":[[
                    np.float64(polygon_aug[0].xx[0]),
                    np.float64(polygon_aug[0].yy[0]),
                    np.float64(polygon_aug[0].xx[1]),
                    np.float64(polygon_aug[0].yy[1]),
                    np.float64(polygon_aug[0].xx[2]),
                    np.float64(polygon_aug[0].yy[2]),
                    np.float64(polygon_aug[0].xx[3]),
                    np.float64(polygon_aug[0].yy[3]),
                    ]],
            })

        count+=1

        for j in range(DATAPOINTS_PER_IMAGE):
            step=random.sample(gen_steps(),1)
            
            if RESIZE:
                if type(step[0]).__name__=='list':
                    step[0].append(final_resize_step)
                    seq = iaa.Sequential(step[0])
                else:
                    step.append(final_resize_step)
                    seq = iaa.Sequential(step)
            else:
                seq = iaa.Sequential(step) 

            img_aug, polygon_aug, bbox_aug = seq(image=img, polygons=poly, bounding_boxes=bbox)

            cv2.imwrite(os.path.join(OUTPUT_DIR,f"img_{count}.jpg"),img_aug)
            data["images"].append({
                "width": img_aug.shape[1],
                "height": img_aug.shape[0],
                "id": count,
                "file_name": f"img_{count}.jpg"
            })

            data["annotations"].append({
                "id": count,
                "image_id": count,
                "category_id": data_from_labstudio["annotations"][i]["category_id"],
                "bbox":[np.float64(bbox_aug[0].x1),np.float64(bbox_aug[0].y1),np.float64(bbox_aug[0].x2),np.float64(bbox_aug[0].y2)],
                "segmentation": [[
                    np.float64(polygon_aug[0].xx[0]),
                    np.float64(polygon_aug[0].yy[0]),
                    np.float64(polygon_aug[0].xx[1]),
                    np.float64(polygon_aug[0].yy[1]),
                    np.float64(polygon_aug[0].xx[2]),
                    np.float64(polygon_aug[0].yy[2]),
                    np.float64(polygon_aug[0].xx[3]),
                    np.float64(polygon_aug[0].yy[3]),
                ]],
            })


            count+=1


    json.dump(data,output_json)



