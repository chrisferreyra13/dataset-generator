import json
from sklearn.model_selection import train_test_split


f=open('output.json','r')
data=json.load(f)
f.close()

images_ids=[]
images_filenames=[]

for img in data["images"]:
    #images_filenames.append(img["file_name"].split('.')[0]+' \n')
    images_ids.append(str(img["id"])+' \n')

x_train, x_test,y_train,y_test=train_test_split(images_ids,images_ids,test_size=0.3, random_state=42)

with open('train.txt','w') as train_file:
    train_file.writelines(x_train)

with open('test.txt','w') as test_file:
    test_file.writelines(x_test)