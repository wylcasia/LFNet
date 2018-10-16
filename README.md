# LFNet
## LFNet: A Novel Bidirectional Recurrent Convolutional Neural Network for Light-field Image Super-resolution
### Yunlong Wang, Fei Liu, Kunbo Zhang, Guangqi Hou, Zhenan Sun, Tieniu Tan

@ARTICLE{LFNet_Wang_2018, 

author={Y. Wang and F. Liu and K. Zhang and G. Hou and Z. Sun and T. Tan}, 

journal={IEEE Transactions on Image Processing}, 

title={LFNet: A Novel Bidirectional Recurrent Convolutional Neural Network for Light-Field Image Super-Resolution}, 

year={2018}, volume={27}, number={9}, pages={4274-4286}, 

doi={10.1109/TIP.2018.2834819}, ISSN={1057-7149}, month={Sept}}

[webpage](https://ieeexplore.ieee.org/document/8356655/)

## Generate Datasets for Training LFNet
* `generate_LFNet_Train_RN.m` for row network of LFNet
* `generate_LFNet_Train_CN.m` for column network of LFNet

## Train From Scratch
* `LFNet_Train_RN_With_IMsF.py` for training row network of LFNet
* `LFNet_Train_CN_With_IMsF.py` for training column network of LFNet

## Utilities
Find some useful functions in `LF2Gif_SceneInFolder_for_LFNet.py`, `algo_evaluation_index_for_LFNet.py ` and so on.





