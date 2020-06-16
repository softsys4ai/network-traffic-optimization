# LSTM-Hyperparameter-Optimization-Network-Traffic

## Dependency 

- `keras`
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `seaborn`
- `matplotlib`
- `yaml`

## Run Instructions

- To run the the code please run the following command from your project 
directory 
```
$ python BO.py [link-number] [lstm-architecture]
```
> To predict on link 3 using 1 layer LSTM:
```
$ python BO.py 3 L1
```
> To predict on link 4 using 4 layer LSTM:
```
$ python BO.py 4 L4
```
- To run optimization on all 8 links for an architetcure please use the following
command from your project directory 
```
$ sh run.sh [lstm-architecture]
```
> To run optimization for 2 layer LSTM:
```
$ sh run.sh L2
```
> To run optimization for 4 layer LSTM:
```
$ sh run.sh L4
```

## Contacts
```
Shahriar: miqbal@email.sc.edu
Bashir: bmohammed@lbl.gov
```

## Acknowledgement

This project is a collaboration with Lawrence Berkeley National Laboratory under DOE Contract number for Deep Learning FP00006145 investigating Large-Scale Deep Learning for Intelligent Networks.
