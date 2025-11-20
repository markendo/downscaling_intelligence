from llava.train.train import train
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*please pass in use_reentrant=True or use_reentrant=False explicitly.*",
    category=UserWarning,
    module="torch.utils.checkpoint"
)

from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.fp16.loss_scaler import LossScaler
import torch

torch.serialization.add_safe_globals([ZeroStageEnum, LossScaler]) # needed for loading checkpoints

if __name__ == "__main__":
    train()
