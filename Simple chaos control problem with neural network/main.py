import torch
from tqdm import tqdm
from model import NeuralNetwork
from loss import loss_fn
from utils import load_checkpoint, save_checkpoint
from plotChaoticSystem import DIM, systemStateVariables, plot_control_chaotic_system
from config import *


# Domain
t_0, t_f = 0, 1
τ = torch.linspace(t_0, t_f - 1e-2, 1_000, requires_grad=True).unsqueeze(-1).to(DEVICE)

# Initialize neural networks
x_mdl = [NeuralNetwork(p1=(t_0, systemStateVariables[i])).to(DEVICE) for i in range(DIM)]
u_mdl = [NeuralNetwork(p1=(t_f, 0)).to(DEVICE) for i in range(DIM)]

# Saving Models
if LOAD_MODEL:
    for i in range(DIM):
        load_checkpoint(x_mdl[i], CHECKPOINT_PATH.format('x', i))
        load_checkpoint(u_mdl[i], CHECKPOINT_PATH.format('u', i))

# Optimizers
x_opt = [torch.optim.AdamW(x_mdl[i].parameters(), lr=LEARNING_RATE) for i in range(DIM)]
u_opt = [torch.optim.AdamW(u_mdl[i].parameters(), lr=LEARNING_RATE) for i in range(DIM)]

# Schedulers
schedulers = list()
schedulers += [
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode='min', 
        factor=LR_SCHEDULER_FACTOR, 
        patience=LR_SCHEDULER_PATIENCE
    )
    for optimizer in x_opt
]
schedulers += [
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode='min', 
        factor=LR_SCHEDULER_FACTOR, 
        patience=LR_SCHEDULER_PATIENCE
    )
    for optimizer in u_opt
]

## Training ##
if __name__ == "__main__":
    loop = tqdm(range(EPOCH))
    for epoch in loop:
        loss = loss_fn(
            τ,
            x_mdl,
            u_mdl,
        )

        for i in range(DIM):
            x_opt[i].zero_grad()
            u_opt[i].zero_grad()

        loss.backward()

        for i in range(DIM):
            x_opt[i].step()
            u_opt[i].step()

        for scheduler in schedulers:
            scheduler.step(loss)

        loop.set_postfix(
            Loss=loss.mean().item(),
            lr=schedulers[0].optimizer.param_groups[0]['lr']
        )

    ## Plot 3D
    plot_control_chaotic_system(x_mdl, u_mdl)

    # Saving Models
    if SAVE_MODEL:
        for i in range(DIM):
            save_checkpoint(x_mdl[i], CHECKPOINT_PATH.format('x', i))
            save_checkpoint(u_mdl[i], CHECKPOINT_PATH.format('u', i))