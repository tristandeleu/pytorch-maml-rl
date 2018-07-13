# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML)
Implementation of Model-Agnostic Meta-Learning (MAML) applied on Reinforcement Learning problems in Pytorch. This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), [Finn et al., 2017](https://arxiv.org/abs/1703.03400)): multi-armed bandits, tabular MDPs, continuous control with MuJoCo, and 2D navigation task.

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
