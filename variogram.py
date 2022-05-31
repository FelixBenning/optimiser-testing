# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import matplotlib.pyplot as plt
import pandas as pd

import functools as func

# %% grid
def carth_grid(dim=3, start=0, end=2, length=3):
    x = (torch.arange(start=start, end=end, step=(end - start) / length),) * dim
    return torch.cartesian_prod(*x)


# %% Import Dataset
ds = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    ),
)

# %% Dataloader
from torch.utils.data import DataLoader

train_data_loader = DataLoader(ds, batch_size=100, shuffle=True)

# %% Parameter normalization
def norm(param_dict):
    return torch.sqrt(
        func.reduce(
            lambda acc, key: acc + torch.sum(torch.square(param_dict[key])),
            param_dict,
            0,
        )
    )


def normalize_params(param_dict):
    """inplace operation"""
    params_norm = norm(param_dict)
    return {name: param / params_norm for name, param in param_dict.items()}


# %% Random Grid
def random_grid(model_factory, grid=carth_grid()):
    model = model_factory()
    origin = {n: p for n, p in model.named_parameters()}
    directions = [
        normalize_params({n: p for n, p in model_factory().named_parameters()})
        for _ in range(grid.size()[1])  # for every dimension
    ]

    def coordinate_system_change(coords):
        return {
            n: p + sum(map(lambda dir, coeff: coeff * dir[n], directions, coords))
            for n, p in origin.items()
        }

    return map(coordinate_system_change, grid)


# %% Pairs
def pairs(unique_elements):
    return (
        (a, b)
        for (k, a) in enumerate(unique_elements)
        for b in unique_elements[k + 1 :]
    )


# %% Plot

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for idx in range(cols * rows):
    sample_idx = torch.randint(len(ds), size=(1,)).item()
    img, label = ds[sample_idx]
    figure.add_subplot(rows, cols, idx + 1)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()

# %%
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(28 * 28, 10, bias=False)

    def forward(self, x):
        return self.dense(torch.flatten(x, 1))


# %% https://github.com/ansh941/MnistSimpleCNN/blob/master/code/models/modelM7.py
class ModelM7(nn.Module):
    def __init__(self):
        super(ModelM7, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, 7, bias=False)  # output becomes 22x22
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 7, bias=False)  # output becomes 16x16
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 7, bias=False)  # output becomes 10x10
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 7, bias=False)  # output becomes 4x4
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(3072, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)

    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits

    def forward(self, x):
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)


# %%
def evaluate(model_factory, params, data, loss):
    model = model_factory()
    model_state = model.state_dict()
    for name, param in params.items():
        model_state[name] = param

    inp, label = data
    return loss(model(inp), label)

# %%
def diff(params1, params2):
    return {name: params1[name]-params2[name] for name in params1}

# %%
def variogram(ModelFactory, data_loader, grid=carth_grid()):
    loss = torch.nn.CrossEntropyLoss()
    return pd.DataFrame.from_records([
        {
            "sqLossDiff": (
                evaluate(ModelFactory, params[0], data, loss)
                - evaluate(ModelFactory, params[1], data, loss)
            )
            ** 2,
            "distance": norm(diff(params[0],params[1])),
        }
        for params, data in zip(pairs(list(random_grid(ModelFactory, grid))), data_loader)
    ])

# %%
variogram(ToyModel, train_data_loader)

#%%
variogram(ModelM7, train_data_loader, carth_grid(dim=2, start=0, end=2, length=5))
# %%
