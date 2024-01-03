# Spatio-Temporal Meta Contrastive Learning

This is the PyTorch implementation by <a href='https://github.com/tjb-tech'>@Jiabin Tang</a> for CL4ST model proposed in this paper:

 >**Spatio-Temporal Meta Contrastive Learning**  
 > Jiabin Tang, Lianghao Xia, Jie Hu, Chao Huang*\
 >*CIKM 2023*

\* denotes corresponding author


In this work, we propose a new spatio-temporal meta contrastive learning framework, called **CL4ST**, to strengthen the robustness and generalization capacity of spatio-temporal modeling. In our **CL4ST**, the meta view generator automatically customizes node- and edge-wise augmentation views for each spatio-temporal graph according to the meta-knowledge of the input graph structure. This approach not only obtains personalized augmentations for every graph but also injects spatio-temporal contextual information into the data augmentation framework. We conduct extensive experiments to evaluate the effectiveness of the **CL4ST** on spatio-temporal prediction tasks, such as traffic forecasting and crime prediction. Comparisons over different datasets show that **CL4ST** outperforms state-of-the-art baselines.

## Environment

Please first clone the repo and install the required environment, which can be done by running the following commands:

```shell
conda env create -n cl4st python=3.8

conda activate cl4st

# Torch with CUDA 11.6
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
# Clone our CL4ST
git clone https://github.com/HKUDS/CL4ST.git
cd CL4ST
# Install required libraries
pip install -r requirements.txt
```



##  Dataset

We utilized three traffic datasets and two crime datasets to evaluate CL4ST: *PEMS4, 7, 8* (Traffic), *NYC, CHI crime* (Crime).

## Examples to run the codes

We could modify configuration at [./config](https://github.com/HKUDS/STExplainer/config) to train or test our model on different datasets. There is an example on PEMS04: 

  - train PEMS4 (note that the "testonly" in configuration file should be 0)

```shell
python train.py --config ./config/CL4ST_pems4.yaml
```

  - test PEMS4 (note that the "testonly" in configuration file should be 1, and there is a corresponding checkpoints at [./results/model](https://github.com/HKUDS/STExplainer/results/model))

```shell
python train.py --config ./config/CL4ST_pems4.yaml
```



## Reference
If you find this work is helpful to your research, please consider citing our paper:
```
@inproceedings{10.1145/3583780.3615065,
author = {Tang, Jiabin and Xia, Lianghao and Hu, Jie and Huang, Chao},
title = {Spatio-Temporal Meta Contrastive Learning},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3615065},
doi = {10.1145/3583780.3615065},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {2412–2421},
numpages = {10},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}
```



## Acknowledgements
The structure of our code is based on [AutoGCL](https://github.com/Somedaywilldo/AutoGCL), [BasicTS](https://github.com/zezhishao/BasicTS). Thank for their work.
