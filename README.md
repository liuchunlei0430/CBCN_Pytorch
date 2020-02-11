## Prepare
Environment: PyTorch (0.4.0), torchvision (0.2.1), tensorboardX, python3, CUDA(8.0)   
get the ImageNet dataset ready    
Install Convolutional Module and Binary Module    
•	cd install    
•	sh install.sh     
•	cd BinActivateFunc_PyTorch    
•	sh install.sh   

## Train and Evaluation
•	Train: python ImageNet.py [dataset_dir] --tensorboard   
•	Evaluation: python ImageNet.py [dataset_dir] --pretrained --tensorboard   


| ResNet18 | Full-precision | CBCN(without centerloss finetune) | CBCN(with centerloss finetune) |
| ------ | ------ | ------ | ------ |
| Top-1 | 69.3 | 61.0 | 61.4 |

Thanks for the code of ORN! Inspired by ORN which already show their powerful ability in within-class rotation-invariance, we also employ similar way to enhance the representative ability which destoryed by the binarization process. More detail can be seen in the install/orn/modules/ORConv.py.    

## Please cite   
  title={Circulant Binary Convolutional Networks: Enhancing the Performance of 1-bit
DCNNs with Circulant Back Propagation},
  author={Liu, ChunLei and Ding, Wenrui and Xia, Xin and Zhang, Baochang and Gu, Jiaxin  and Liu, Jianzhuang and Ji, Rongrong and David, Doermann },
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}   
@ inproceedings{Zhou2017ORN,
    author = {Zhou, Yanzhao and Ye, Qixiang and Qiu, Qiang and Jiao, Jianbin},
    title = {Oriented Response Networks},
    booktitle = {CVPR},
    year = {2017}
}

