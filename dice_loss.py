import torch
import torch.nn as nn
from torch.autograd import Variable

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.softmax = torch.nn.Softmax(1)

    def forward(self, inputs, labels, device, epsilon=1e-10, from_label=1, to_label=-1):
        # the labels are not one-hot encoded yet
        num_classes = inputs.shape[1]
        #print(torch.unique(labels.squeeze(1)))
        labels_1_hot = torch.eye(num_classes)[labels.squeeze(1)].to(device)
        labels = labels_1_hot.permute(0, 3, 1, 2)
        prediction = self.softmax(inputs).to(device)

        intersection = prediction * labels
        intersec_per_img_per_lab = torch.sum(intersection, dim=[2, 3])

        l = torch.sum(prediction, dim=[2, 3])
        r = torch.sum(labels, dim=[2, 3])

        dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)
        
        loss = 1 - torch.mean(dices_per_subj[:, from_label:])

        return loss

    