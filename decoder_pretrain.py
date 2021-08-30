from decoder import Decoder
from encoder import Encoder
import torch.nn as nn
import numpy as np

import pdb

#TODO: in separate file
class DebugLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pdb.set_trace()
        return x

class DecoderPretrainNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_channels = config["n_channels"]
        n_classes = config["n_classes"]

        self.enc = Encoder(config)

        # freeze encoder weights
        for param in self.enc.parameters():
            param.requires_grad = False

        self.dec = Decoder(config, self.enc)
        dec_blocks = config["n_dec_blocks"]
        inter_channels = np.array(config["no_filters"])[-dec_blocks-1]
        self.g2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1)
                                )

    def forward(self, x):
        x = self.enc(x)
        self.dec.enc_outputs = self.enc.enc_outputs

        x = self.dec(x)
        x = self.g2(x)

        return x
