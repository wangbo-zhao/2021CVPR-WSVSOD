# WSVSOD
🔥🔥🔥Code for Paper in CVPR2021, [Weakly Supervised Video Salient Object Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Weakly_Supervised_Video_Salient_Object_Detection_CVPR_2021_paper.pdf) Wangbo Zhao, Jing Zhang, Long Li, Nick Barnes, Nian Liu, Junwei Han.
🔥🔥🔥 

![](https://github.com/wangbo-zhao/WSVSOD/blob/main/image.png?raw=true])
## Dataset
We provide the proposed scribble annotations for DAVIS and DAVSOD in [Google Drive](https://drive.google.com/drive/folders/1gZZQ_JgwcoH6oHMBCcEZxv3iBQrOAP36?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/11zN_MuYaV7l_p36Ba-FO2Q)(86rd). [SSOD](https://github.com/JingZhang617/Scribble_Saliency) provides scribble annotations for DUTS.

## Code
First,
```
python pretrain.py
```
to pretrain the model on DUTS.

Then, 
```
python finetune.py
```
to finetune the model on DAVSOD and DAVIS.

## Pretrained Models
We provide the pretrained "Our" model in [Baidu Drive](https://pan.baidu.com/s/14X4pknWrnP_KQ9oG1jPSBA)(30ja).

## Saliency Map
Here, we provide saliency maps generated from "Our" in the paper. [Baidu Drive](https://pan.baidu.com/s/1k8cfCBM4g1flM_dZ1OSt5A)(iv0d).



## Evaluation
We borrow the evaluation code from [DAVSOD](https://github.com/DengPingFan/DAVSOD), which is based on MATLAB.

## Acknowledgement

[Weakly-Supervised Salient Object Detection via Scribble Annotations](https://github.com/JingZhang617/Scribble_Saliency)  
[Shifting More Attention to Video Salient Object Detection](https://github.com/DengPingFan/DAVSOD)  
[Paying Attention to Video Object Pattern Understanding](https://github.com/wenguanwang/AGS)  

