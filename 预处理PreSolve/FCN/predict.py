from net.fcn import fcn_net
from PIL import Image
import numpy as np
import random
import copy
import os

UNIFIED_HEIGHT = 256
UNIFIED_WIDTH = 256
BATCH_SIZE = 1

weights_path="D:/jupyter/FCN/logs/ep021-loss0.187-val_loss0.177.h5"
predict_path="F:/我的资源/data/predict"

model = fcn_net(BATCH_SIZE)
model.load_weights(weights_path)

file_name = os.listdir(predict_path+"/input/")
j = 0
file_num = len(file_name)
img_name = os.listdir(predict_path+"/input/"+file_name[j]+'/')
i = 0
img_num = len(img_name)

for i in range(img_num):

    img = Image.open(predict_path+"/input/"+file_name[j]+'/'+img_name[i])
    input_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((UNIFIED_WIDTH,UNIFIED_HEIGHT))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1,UNIFIED_WIDTH,UNIFIED_HEIGHT,3)
    pr = model.predict(img)[0]
    pr = pr.reshape(UNIFIED_WIDTH,UNIFIED_HEIGHT,3)
    pr = pr*255

    out_img = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h))
    
    out_img.save(predict_path+"/result/"+file_name[j]+'/'+img_name[i])
    out_img.show()
    print('{}/{} finished!\n'.format(file_name[j],img_name[i]))
