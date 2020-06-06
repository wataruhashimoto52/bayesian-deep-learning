# Experiments on Bayesian Neural Networks and their uncertainies.

## Algorithms
* Monte-Carlo Dropout (MC Dropout)
* Stochastic Gradient Langevin Dynamics (SGLD)
* preconditioned Stochastic Gradient Langevin Dynamics (pSGLD)
* Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)
* Variational Online Gauss Newton (VOGN)
* Kronecker-Factored Approximate Curvature (KFAC)
* noisy KFAC
* Stochastic Weight Averaging (SWA)
* SWA-Gaussian (SWAG)


## Dataset
* ImageNet
* CIFAR-10
* STL-10
* UCI 3D Road data


## How to run.

```

```

## 実験の全体像
* データセットを用意し、動かせるところまでいく。
* 元論文＋周辺論文を読む。
* 各データセットにおける実験で比較したい評価指標を明らかにし、それを求める。
  * 現在だと、Toy(回帰)で不確実性表示、回帰(3droad.mat)、分類(CIFAR-10)
  * 適宜必要ならばプロットや表にまとめる。
  * 今考えているのは、
    * 回帰モデルにおける2乗誤差と不確実性。不確実性と誤差が相関していると嬉しい。
    * 分類モデルにおける分類精度。さらに、分類問題において不確実性はどういうふうに得られるのかを知りたい。
* READMEに実行手順とDockerでの実行方法をまとめ、誰でもすぐに実行できるようにする。


## Reference

### Papers 


### Experimental Codes
* https://github.com/wjmaddox/swa_gaussian 
* https://github.com/team-approx-bayes/dl-with-bayes
* https://github.com/MFreidank/pysgmcmc 