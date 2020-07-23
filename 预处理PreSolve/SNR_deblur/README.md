```bibtex
@inproceedings{tao2018srndeblur,
  title={Scale-recurrent Network for Deep Image Deblurring},
  author={Tao, Xin and Gao, Hongyun and Shen, Xiaoyong and Wang, Jue and Jia, Jiaya},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```
***
## training
+ model：（百度云）下载到checkpoints/color/checkpoints
+ data：根据datalist.txt放置到./training_set（自建）
+ datalist.txt：记录data对路径，其中每一行前一个是期望输出（TACO原图），后一个是输入（注水后初步去水，即WaterCo-2.0），用空格隔开
+ file.m：用于缠上datalist.txt
+ run_model.py：
    1. train时如果不用gpu则gpu=-1
    2. 其余参照原作者的README.md（放在了./models）
+ model.py：主要修改train函数
    1. global_step = tf.Variable(initial_value=420, dtype=tf.int32, trainable=False)中的initial_value改为断点步数（=0则从头开始）
    2. ckpt_name = model_name + '-' + str(420)中str()改为断点步数（如果从头开始就把相关202-205行都注释掉）
+ logs.txt：记录训练过程，可根据loss选择合适的model

## testing
+ run_model.py：
    1. 将args.phase = 'train'注释掉
    2. 设置input_path和output_path
+ model.py：主要修改test函数
    1. self.load(sess, checkpoint_path, step=420)中step改为断点步数

***
_如出现问题，大概率是路径问题！！！_
