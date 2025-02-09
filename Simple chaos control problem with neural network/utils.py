import torch
from config import (
    DEVICE,
)

def save_checkpoint(model:torch.nn.Module, filename:str) -> None:
    """ Saving Checkpoint... """
    torch.save(model.state_dict(), filename)

def load_checkpoint(model:torch.nn.Module, filename:str) -> None:
    """ Loading pretrained weights... """
    model.load_state_dict(torch.load(filename, map_location=DEVICE, weights_only=True), strict=False)