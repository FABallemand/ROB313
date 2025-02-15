import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter:
    """
    Compute running average
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NTXentLoss(nn.Module):

    def __init__(self, temperature):
        super().__init__()
        self.temp = temperature

    def forward(self, x):
        assert len(x.size()) == 2

        # Cosine similarity
        xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
        xcs[torch.eye(x.size(0)).bool()] = float("-inf")

        # Ground truth labels
        target = torch.arange(x.size(0), device=x.device)
        target[0::2] += 1
        target[1::2] -= 1

        # Standard cross-entropy loss
        loss = F.cross_entropy(xcs / self.temp, target, reduction="mean")
        return loss, {"loss": loss, "ntxent": loss}


class AutoassociativeLoss(nn.Module):

    def __init__(self, lmbda, temperature):
        super().__init__()
        self.lmbda = lmbda
        self.mse = nn.MSELoss()
        self.ntxent = NTXentLoss(temperature)

    def forward(self, x, z, target):
        mse_loss = self.mse(x, target)
        ntxent_loss = self.ntxent(z)
        loss = self.lmbda * mse_loss + (1 - self.lmbda) * ntxent_loss
        return loss, {"loss": loss, "mse": mse_loss, "ntxent": ntxent_loss}