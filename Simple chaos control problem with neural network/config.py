import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 4.5e-4
LR_SCHEDULER_FACTOR = 0.3
LR_SCHEDULER_PATIENCE = 500
EPOCH = 10_000

CHECKPOINT_PATH = "ChaosControl_{}{}.pth"

LOAD_MODEL = False
SAVE_MODEL = True