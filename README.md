# Text2Data-private
This is the official Repository of [Tex2Data (AAAI 2025)](https://arxiv.org/pdf/2402.10941), which is a training strategy designed for low-resource data generation that can be seaminglessly adapted to almost any generative models.

# Method overview
Text2Data is a training framework for low-resource data generation. It can be seamlessly adapted to the training process of almost any generative models (see [Implementation instruction](#implementation-instruction)). 

<img src="https://github.com/shi-yu-wang/Text2Data-private/blob/main/model.png" width="450" height="300">

We initially utilize all data (blue module) in the dataset, and treat them as unlabelled data to pre-train the generative model to discern the overall data distribution while the optimal set of model parameters $$\Theta$$ is obtained. Then we finetune this generative model using only labelled data (e.g., data-text pairs, red module) to achieve desired model control. Crucially, we ensure that the parameters during finetuning using labelled data (i.e., $$\Theta'$$) remain in close proximity to those established during the initial training using unlabelled data (i.e., $$\Theta$$).

# Required Python packages
This method only needs `PyTorch` and `collections`.

# Repo files and important arguments
## Files in repo
- `constraint.py`: This file contains implementation program of lexicographic optimization.
- `text2data.py`: This file includes the function users use to implement Text2Data.

## Important arguments in text2data.py
We call function `run_step()` in `text2data.py` to implement Text2Data algorithm, which involves the following arguments:
- `optimizer`: "Constrain" object defined in `constraint.py`.
- `g_constraint`: Minimum value of unconditional loss (see Eq. (5) in [paper](https://arxiv.org/pdf/2402.10941)). It is the minimum value of unconditional loss over all input data.
- `model`: The model that is being trained. It usually contains two arguments for training purpose: (1) Input data (`data` arguments below, e.g., input data of forward process to train a diffusion model) and (2) Conditions that controllable generation relies on (i.e., `cond` arguments below). 
- `cond`: Conditions that the model relies on, e.g., descriptive texts, molecular properties, etc.
- `data`: Input data of the model.
- `label`: Labels/ground-truth data that the model predicts/generates (e.g., standard Gaussian in diffusion loss). Usually needed to compute loss.
- `loss_fn`: Loss function that is used to compute loss, such as `torch.nn.functional.mse_loss`.

## Important arguments in constraint.py
`constraint.py` contains the implementation algorithm of lexicographic optimization. It involves the following arguments:
- `params`: Model parameters. Usually it is `model.parameters()`, where `model` is user's model trained with Text2Data.
- `base_optimizer`: Base optimizer, such as `torch.optim.Adam` or `torch.optim.AdamW`.
- `alpha`: Hyperparameter to be tuned. Default is 1. See Eq. (8) in [paper](https://arxiv.org/pdf/2402.10941).
- `beta`: Hyperparameter to be tuned. Default is 1. See Eq. (8) in [paper](https://arxiv.org/pdf/2402.10941).
- `**kwargs`: other arguments passed to `base_optimizer`, such as `lr` and `weight_decay`.

# Implementation instruction
It is simple to utilize our algorithm:

1. Download `constraint.py` and `text2data.py` and put them in the folder where you training code is.
2. Package your training code in the following manner:
```bash

import torch
import torch.nn.functional as F # You may import any other packages needed for your training process.

from constraint import Constraint # required
from text2data import run_step # required

model = ... # Define your model here.

g_constraint = ... # g_constraint is a scalar, usually the lowest loss obtained during the pre-training process. You can relax g_constraint by multiplying it by a hyperparameter.

optimizer = Constraint(model.parameters(), torch.optim.Adam, lr=1e-3) # Package your optimizer using Constraint in constraint package. Please make sure you have defined your model above. You may change/add other parameters to optimizer in addition to "lr". You can switch to other optimizers, such as torch.optim.AdamW.

for epoch in epoches:
    cond, x, label in enumerate(data_loader): # Here assume dataloader gives us condition (i.e., cond), data (i.e., x) and label. Condition and data serve as the input to the model (e.g., original data and data properties in classifier-free diffusion models). Label will be used to compute loss function with the model output (e.g., standard Gaussian in diffusion loss).
        run_step(optimizer, g_constraint, model, cond, x, label, F.mse_loss) # This will automatically take gradients, run text2data algorithm and update parameters. You can user other loss functions.
```

# Citation
```bash
@article{wang2024text2data,
  title={Text2Data: Low-Resource Data Generation with Textual Control},
  author={Wang, Shiyu and Feng, Yihao and Lan, Tian and Yu, Ning and Bai, Yu and Xu, Ran and Wang, Huan and Xiong, Caiming and Savarese, Silvio},
  journal={arXiv preprint arXiv:2402.10941},
  year={2024}
}
```
