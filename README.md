# Model X-ray

The official implementation of our ACM Multimedia 2024 paper "Model X-ray: Detecting Backdoored Models via Decision Boundary".[[Paper](https://arxiv.org/abs/2402.17465v2)] 

![Backdoor Detection](https://img.shields.io/badge/Backdoor-Detction-yellow.svg?style=plastic)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.11.0](https://img.shields.io/badge/pytorch-1.11.0-orange.svg?style=plastic)

## Abstract
Backdoor attacks pose a significant security vulnerability for deep neural networks (DNNs), enabling them to operate normally on clean inputs but manipulate predictions when specific trigger patterns occur. Currently, post-training backdoor detection approaches often operate under the assumption that the defender has knowledge of the attack information, logit output from the model, and knowledge of the model parameters. In contrast, our approach functions as a lightweight diagnostic scanning tool offering interpretability and visualization. By accessing the model to obtain hard labels, we construct decision boundaries within the convex combination of three samples. We present an intriguing observation of two phenomena in backdoored models: a noticeable shrinking of areas dominated by clean samples and a significant increase in the surrounding areas dominated by target labels. Leveraging this observation, we propose Model X-ray, a novel backdoor detection approach based on the analysis of illustrated two-dimensional (2D) decision boundaries. Our approach includes two strategies focused on the decision areas dominated by clean samples and the concentration of label distribution, and it can not only identify whether the target model is infected but also determine the target attacked label under the all-to-one attack strategy. Importantly, it accomplishes this solely by the predicted hard labels of clean inputs, regardless of any assumptions about attacks and prior knowledge of the training details of the model. Extensive experiments demonstrated that Model X-ray has outstanding effectiveness and efficiency across diverse backdoor attacks, datasets, and architectures. Besides, ablation studies on hyperparameters and more attack strategies and discussions are also provided.
## Deploy Model X-ray on BackdoorBench-v2 Codebase
### Setup
- **Get Model X-ray**
```shell 
git clone https://github.com/SuYanghao/Model_X-ray.git
cd Model_X-ray
```
- **Get BackdoorBench-v2**\
*Merge Model X-ray into the [BackdoorBench-v2](https://github.com/SCLBD/BackdoorBench) codebase*
```shell 
git clone -b v2 https://github.com/SCLBD/BackdoorBench.git
rsync -av BackdoorBench-v2-merge/BackdoorBench/
cd BackdoorBench
sh ./sh/install.sh
mkdir record
mkdir data
mkdir data/cifar10
mkdir data/cifar100
mkdir data/gtsrb
mkdir data/tiny
```


### Quick Start
- **Train a Backdoor Model**
```
python ./attack/badnet.py --yaml_path ../config/attack/badnet/cifar10.yaml --dataset cifar10 --dataset_path ../data --model preactresnet18 --save_folder_name cifar10_preactresnet18_badnet
```

- **Model X-ray Evaluation**
```
python analysis/modelxray.py --result_file cifar10_preactresnet18_badnet --yaml_path ./config/visualization/modelxray.yaml --dataset cifar10 --model preactresnet18
```

For guidance on conducting more evaluations, such as using different attacks, datasets, and model architectures, please refer to [BackdoorBench](https://github.com/SCLBD/BackdoorBench).

## BibTeX 
If you find Model X-ray both interesting and helpful, please consider citing us in your research or publications:
```bibtex
@inproceedings{su2024model,
  title={Model X-ray: Detecting Backdoored Models via Decision Boundary},
  author={Su, Yanghao and Zhang, Jie and Xu, Ting and Zhang, Tianwei and Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={10296--10305},
  year={2024}
}
```
## Acknowledge
```bibtex
@inproceedings{backdoorbench,
  title={BackdoorBench: A Comprehensive Benchmark of Backdoor Learning},
  author={Wu, Baoyuan and Chen, Hongrui and Zhang, Mingda and Zhu, Zihao and Wei, Shaokui and Yuan, Danni and Shen, Chao},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

