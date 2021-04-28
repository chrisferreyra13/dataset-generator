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

final_size_step=iaa.Resize({"width": 640, "height": "keep-aspect-ratio"})


# Steps to apply
def gen_steps(num_of_transforms):

    final_steps=[]
    STEPS=[]
    for i in range(num_of_transforms):
        STEPS=[
            iaa.Rotate((-50, 50)),
            iaa.Affine(shear=(-50,50)),
            iaa.ShearX(((-20, 20))),
            iaa.ShearY(((-20, 20))),
            iaa.Affine(scale=(0.5, 1.5)),
            iaa.Affine(translate_px={"x": (1, random.randint(300,800)), "y": (1, random.randint(300,800))}),
            [
                iaa.Affine(translate_px={"x": (1, random.randint(300,800)), "y": (1, random.randint(300,800))}),
                iaa.Affine(shear=(-50,50)),
            ],
            [
                iaa.Affine(translate_px={"x": (1, random.randint(300,800)), "y": (1, random.randint(300,800))}),
                iaa.Affine(shear=(-50,50)),
            ],
        ]
        step=random.sample(STEPS,1)
        final_steps.append(step)
    

    return final_steps


# read json
JSON_FILE='result.json'
OUTPUT_DIR='./dataset'

os.makedirs(OUTPUT_DIR,exist_ok=True)

f=open(JSON_FILE,'r')
data_from_labstudio=json.load(f)
f.close()

with open('output.json','w') as output_json:
    num_images=len(data_from_labstudio["images"])
    count=num_images

    for i in range(num_images):
        segment=data_from_labstudio["annotations"][i]["segmentation"][0]
        poly = [ia.Polygon([
            (segment[0],segment[1]), 
            (segment[2],segment[3]), 
            (segment[4],segment[5]), 
            (segment[6],segment[7])]
            )]
        
        img=cv2.imread(data_from_labstudio["images"][i]["file_name"])

        for j in range(DATAPOINTS_PER_IMAGE):
            step=random.sample(gen_steps(DATAPOINTS_PER_IMAGE),1)
            if RESIZE:
                seq = iaa.Sequential(step.append(final_size_step))
            else:
                seq = iaa.Sequential(step)

            img_aug, polygon_aug = seq(image=img, polygons=poly)

            cv2.imwrite(os.path.join(OUTPUT_DIR,f"img_{count}.jpg"),img_aug)
            data_from_labstudio["images"].append({
                "width": img_aug.shape[1],
                "height": img_aug.shape[0],
                "id": count,
                "file_name": f"img_{count}.jpg"
            })

            data_from_labstudio["annotations"].append({
                "id": count,
                "image_id": count,
                "category_id": 0,
                "segmentation": [
                    [
                    np.float64(polygon_aug[0].xx[0]),
                    np.float64(polygon_aug[0].yy[0]),
                    np.float64(polygon_aug[0].xx[1]),
                    np.float64(polygon_aug[0].yy[1]),
                    np.float64(polygon_aug[0].xx[2]),
                    np.float64(polygon_aug[0].yy[2]),
                    np.float64(polygon_aug[0].xx[3]),
                    np.float64(polygon_aug[0].yy[3]),
                    ]
                ],
            })


            count+=1


    json.dump(data_from_labstudio,output_json)



