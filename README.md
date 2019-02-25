Use:

pytorch-maml-rl-master lsharp$ python main.py --env-name 2DNavigation-v0 --num-workers 8 --fast-lr 0.1 --fast-batch-size 20 --meta-batch-size 30  --num-batches 1000 --gamma 0.99

Customized MAML for navigation in dense crowds. 

```
@article{DBLP:journals/corr/FinnAL17,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {Model-{A}gnostic {M}eta-{L}earning for {F}ast {A}daptation of {D}eep {N}etworks},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```
