import torch.nn as nn
import torch


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        self.class_weights = class_weights

    def forward(self, pred, target):
        weights = torch.sum(self.class_weights * target, dim=3)

        # For weighted error
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = torch.nn.functional.cross_entropy(pred, target)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        return torch.mean(weighted_losses)
