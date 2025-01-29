import shutil
import json
import random
import numpy as np

from einops import rearrange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_segmentation_masks

from torchmetrics.utilities.compute import _safe_divide

from torch_uncertainty.datasets import MUAD
from torch_uncertainty.metrics.classification import MeanIntersectionOverUnion

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Training parameters
batch_size = 10
num_workers = 2
learning_rate = 1e-3
weight_decay = 2e-4
lr_decay_epochs = 20
lr_decay = 0.1
nb_epochs = 50

train_transform = v2.Compose(
    [
        v2.Resize(size=(256, 512), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64,
                "others": None,
            },
            scale=True,
        ),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = v2.Compose(
    [
        v2.Resize(size=(256, 512), antialias=True),
        v2.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.int64,
                "others": None,
            },
            scale=True,
        ),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_set = MUAD(root="./data", target_type="semantic", version="small", split="train", transforms=train_transform, download=True)
val_set = MUAD(root="./data", target_type="semantic", version="small", split="val", transforms=val_transform, download=True)
test_set = MUAD(root="./data", target_type="semantic", version="small", split="test", transforms=val_transform, download=True)

train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper.

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References:
        https://arxiv.org/abs/1606.02147

    Args:
        dataloader (``data.Dataloader``): A data loader to iterate over the
            dataset.
        num_classes (``int``): The number of classes.
        c (``int``, optional): AN additional hyper-parameter which restricts
            the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()
        # Flatten label
        flat_label = label.flatten()
        flat_label = flat_label[flat_label != 255]

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    return 1 / (np.log(c + propensity_score))

print("Computing class weights...")
print("(this can take a while depending on the dataset size)")
class_weights = enet_weighing(train_loader, 22)
class_weights = torch.from_numpy(class_weights).float().cuda()
print("Class weights:", class_weights)

for i, muad_class in enumerate(train_set.classes):
    class_name = muad_class.name
    train_id = muad_class.id
    print(f"Class: {class_name:35} | Train ID: {train_id:2} | Weight: {class_weights[i]}")


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.resize(x1, size=[2*x1.size()[2],2*x1.size()[3]],
                          interpolation=v2.InterpolationMode.BILINEAR)
        else:
            x1 = self.up(x1)

        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    # Please note that we have added dropout layer to be abble to use MC dropout
    def __init__(self, classes):
        super().__init__()
        self.inc = InConv(3, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 32)
        self.dropout = nn.Dropout2d(0.1)
        self.outc = OutConv(32, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.dropout(x)
        x = self.up4(x, x1)
        x = self.dropout(x)
        return self.outc(x)


def train(model, data_loader, optim, criterion, metric, iteration_loss=False):
    model.train()
    epoch_loss = 0.0
    metric.reset()
    for step, batch_data in enumerate(data_loader):
        # Get the inputs and labels
        img = batch_data[0].cuda()
        labels = batch_data[1].cuda()

        # Forward propagation
        logits = model(img)
        flatten_logits = rearrange(logits, "b c h w -> (b h w) c")
        flatten_labels = labels.flatten()

        valid_mask = flatten_labels != 255

        # Loss computation
        loss = criterion(flatten_logits[valid_mask], flatten_labels[valid_mask])

        # Backpropagation
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Keep track of loss for current epoch
        epoch_loss += loss.item()

        # Keep track of the evaluation metric
        metric.update(flatten_logits[valid_mask].detach(), flatten_labels[valid_mask].detach())

        if iteration_loss:
            print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

    # Compute IoU per class
    tp, fp, _, fn = metric._final_state()
    iou_per_class = _safe_divide(tp, tp + fp + fn, zero_division=float("nan"))

    return epoch_loss / len(data_loader), iou_per_class, metric.compute().item()


def test(model, data_loader, criterion, metric, iteration_loss=False):
    model.eval()
    epoch_loss = 0.0
    metric.reset()
    for step, batch_data in enumerate(data_loader):
        # Get the inputs and labels
        img = batch_data[0].cuda()
        labels = batch_data[1].cuda()

        with torch.no_grad():
            # Forward propagation
            logits = model(img)
            flatten_logits = rearrange(logits, "b c h w -> (b h w) c")
            flatten_labels = labels.flatten()

            valid_mask = flatten_labels != 255

            # Loss computation
            loss = criterion(flatten_logits[valid_mask], flatten_labels[valid_mask])

        # Keep track of loss for current epoch
        epoch_loss += loss.item()

        # Keep track of evaluation the metric
        metric.update(flatten_logits[valid_mask], flatten_labels[valid_mask])

        if iteration_loss:
            print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

    # Compute IoU per class
    tp, fp, _, fn = metric._final_state()
    iou_per_class = _safe_divide(tp, tp + fp + fn, zero_division=float("nan"))

    return epoch_loss / len(data_loader), iou_per_class, metric.compute().item()


def save_checkpoint(
        checkpoint, is_best, path="checkpoint.pth.tar",
        best_path="checkpoint_best.pth.tar"):
    torch.save(checkpoint, path)
    if is_best:
        shutil.copyfile(path, best_path)


if __name__ == "__main__":
    # job_id = sys.argv[1]
    # save_path = f"train_res/{job_id}"

    # Intialize U-Net
    n_classes = 22
    net = UNet(n_classes).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay)
    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion combines LogSoftMax and NLLLoss.
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    metric = MeanIntersectionOverUnion(n_classes).to(device) # ignore_index=21???

    # Run training and track with wandb
    train_loss_history = []
    val_loss_history = []
    train_miou_history = []
    val_miou_history = []
    best_loss = float("inf") # Best loss
    # nb_epochs = 2 # FOR TESTING PURPOSE ONLY
    for epoch in range(1, nb_epochs+1):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Train one epoch
        train_loss, iou_per_class, train_metric = train(net, train_loader, optimizer, criterion, metric, iteration_loss=False)
        train_loss_history.append(train_loss)
        train_miou_history.append(train_metric)

        # Validation
        val_loss, iou_per_class, val_metric = test(net, val_loader, criterion, metric, iteration_loss=False)
        val_loss_history.append(val_loss)
        val_miou_history.append(val_metric)

        # Learning rate scheduler step
        lr_scheduler.step(val_loss)

        # Save model
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint(net.state_dict(), is_best)

        # Logging
        if (epoch % 10) == 0:
            print(
                f"[Epoch {epoch:2}/{nb_epochs}]"
                f"Train: loss={train_loss:.3f}, metric={train_metric:.3f} |"
                f"Validation: loss={val_loss:.3f}, metric={val_metric:.3f}"
            )

    # Save model
    torch.save(net.state_dict(), "model.pth")
    with open("train_loss_history.txt", "w") as f:
        f.write(json.dumps(train_loss_history))
    with open("val_loss_history.txt", "w") as f:
        f.write(json.dumps(val_loss_history))
    with open("train_miou_history.txt", "w") as f:
        f.write(json.dumps(train_miou_history))
    with open("val_miou_history.txt", "w") as f:
        f.write(json.dumps(val_miou_history))