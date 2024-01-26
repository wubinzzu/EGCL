# EGCL
This is our Tensorflow implementation for our EGCL 2024 paper and a part of baselines:

>Bin Wu, Bo Zhang &Yihao  Tian &Jing Liang & Yangdong Ye. EGCL: An Effective and Efficient Graph Contrastive Learning Framework for Social Recommendation, IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, SUBMISSION 2024.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.14.0
* numpy == 1.16.4
* scipy == 1.3.1
* pandas == 0.17

## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.
NOTE: The cpp implementation is much faster than python.**

## Examples to run BPRMF:
run [main.py](./main.py) in IDE or with command line:
```
python main.py
```

NOTE :   
(1) the duration of training and testing depends on the running environment.  
(2) set model hyperparameters on .\conf\BPRMF.properties  
(3) set NeuRec parameters on .\NeuRec.properties  
(4) the log file save at .\log\Ciao\  

## Dataset
We provide Ciao dataset.
  * .\dataset\Ciao.rating and Ciao.uu
  *  Each line is a user with her/his positive interactions with items: userID \ itemID \ ratings.
  *  Each user has more than 0 associated actions.

## Baselines
The list of available models in BPRMF along with their paper citations, are shown below:

| General Recommender | Paper                                                        |
| ------------------- | ------------------------------------------------------------ |
| BPRMF               | Steffen Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009. |
| LightGCN            | Xiangnan He, et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020. |
