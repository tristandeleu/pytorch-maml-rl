# On the Robustness of Context- and Gradient-based Meta-Reinforcement Learning Algorithms

For installing MuJoCo refer [here](https://github.com/openai/mujoco-py).

## Setting the environment

Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```bash
virtualenv venv --python=python3.7
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce results

Parameter --type denotes the amount of quaters included during training.
Parameter --num-steps denotes the amount of gradient steps to take.

#### Training
You can use the [`train.py`](train.py) script in order to run reinforcement learning experiments with MAML. Note that by default, logs are available in [`train.py`](train.py) but **are not** saved (eg. the returns during meta-training). For example, to run the script on HalfCheetah-Vel:
```
python train.py --config configs/maml/ant-goal.yaml --output-folder maml-ant-goal --seed 1 --num-workers 4 --type 2
```

#### Testing
Once you have meta-trained the policy, you can test it on the same environment using [`test.py`](test.py):
```
python test.py --config maml-ant-goal/config.json --policy maml-ant-goal/policy.th --output maml-ant-goal/results.npz --seed 1 --meta-batch-size 20 --num-batches 10  --num-workers 4 --num-steps 10
