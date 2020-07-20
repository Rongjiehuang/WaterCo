#WaterCo#

# By Super 你是哪一种垃圾 #


1. fcn.py：FCN网络构架，输入256*256*3，输出256*256*3

2. train.py：训练，注意更改输入输出图像路径
+ 我的储存方式是train一个文件夹，下面有input和ouput，valid同理
+ train的input放了WaterCO的batch_1_new-batch_7_new（但为了方便输入文件夹名称统一为batch_1-batch_7），output放了TACO的batch_1-batch_7
+ valid的input放了WaterCO的batch_8_new-batch_10_new（但为了方便输入文件夹名称统一为batch_8-batch_10），output放了TACO的batch_8-batch_10
+ 稍微改过参数后有部分注释没有修改，请谅解

3. predict.py：预测，注意更改输入输出图像路径
+ 我的储存方式是predict一个文件夹，下面有input和result
+ train的input放了WaterCO的batch_11_new-batch_15_new

***logs里面是最近一次的训练权重***
