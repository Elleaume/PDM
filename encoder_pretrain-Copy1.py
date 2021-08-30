import numpy as np
import torch.nn as nn
import pdb

from encoder import Encoder


class EncoderPretrainNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        output_size_squared = np.prod(np.array(config["resize_size"]) // (1 << (len(config["no_filters"]) - 1)))
        self.inter_ch = int(output_size_squared*config["no_filters"][-1])

        self.enc = Encoder(config)
        self.g1 = nn.Sequential(nn.Flatten(),
                                nn.Linear(self.inter_ch, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 128)
                                )

    def forward(self, x):
        x = self.enc(x)
        x = self.g1(x)
        return x
