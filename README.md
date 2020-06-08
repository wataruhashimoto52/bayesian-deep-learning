# Experiments on Inference Algorithms in Bayesian Neural Networks.

## Experiments
* Classification (CIFAR-10, ImageNet, etc...)
* Regression (UCI 3D Road data)

## Algorithms
* Adam (Non-Bayesian)
* Stochastic Gradient Langevin Dynamics (SGLD)
* preconditioned Stochastic Gradient Langevin Dynamics (pSGLD)
* Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)
* Kronecker-Factored Approximate Curvature (KFAC)
* noisy KFAC
* Variational Online Gauss Newton (VOGN)


## How to run.

* (nvidia-docker) Setup

```
$ docker build -t bnns:0.1 .
$ docker run --rm --gpus all -it bnns:0.1 /bin/bash
```

* Classification (CIFAR-10)

```
$ python src/main_classification.py --download True --config <path/to/config>
```

* Regression (UCI 3D Road data) 

```
$ python src/main_regression.py --config <path/to/config> --log_name hoge
``` 

or 

```
$ python src/main_regression.py --optim_name <optimizer object name> --log_name hoge
```


## Reference

### Papers 
* [Bayesian Learning via Stochastic Gradient Langevin Dynamics](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf)
* [Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks](https://arxiv.org/abs/1512.07666)
* [Stochastic Gradient Hamiltonian Monte Carlo](https://arxiv.org/abs/1402.4102)
* [Noisy Natural Gradient as Variational Inference](https://arxiv.org/abs/1712.02390)
* [Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam](https://arxiv.org/abs/1806.04854)
* [Practical Deep Learning with Bayesian Principles](https://arxiv.org/abs/1906.02506)

### Experimental Codes
* https://github.com/wjmaddox/swa_gaussian 
* https://github.com/team-approx-bayes/dl-with-bayes
* https://github.com/MFreidank/pysgmcmc 
