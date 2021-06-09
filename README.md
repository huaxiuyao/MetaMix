# MetaMix

## About
Source code of the paper [Improving Generalization in Meta-learning via Task Augmentation](https://arxiv.org/abs/2007.13040). This code is built upon the pytorch implementation of few-shot learning [few-shot](https://github.com/oscarknagg/few-shot) and the implementation of [MAML++](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch), [MR-MAML](https://github.com/mingzhang-yin/Meta-learning-without-memorization).


If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{yao2021improving,
  title={Improving Generalization in Meta-learning via Task Augmentation},
  author={Yao, Huaxiu and Huang, Longkai and Zhang, Linjun and Wei, Ying and Tian, Li and Zou, James and Huang, Junzhou and Li, Zhenhui},
  booktitle={Proceeding of the Thirty-eighth International Conference on Machine Learning},
  year={2021} 
}
```

## Data
We have put the related datasets in [Google Drive](https://drive.google.com/drive/folders/1nKZQBV-NVwnvHOGFxZovb4AJyWp7_GTz?usp=sharing)

## Usage
### Dependence
* python 3.*
* Pytorch 1.17+
* Tensorflow 1.15 (for Pose only)

### Drug
Please see the bash file in /Drug for hyperparameter settings

### Pose
Please see the bash file in /Pose for hyperparameter settings

### miniImagenet
Please see the bash file in /miniImagenet for hyperparameter settings

### Omniglot
Please see the bash file in /omniglot for hyperparameter settings
