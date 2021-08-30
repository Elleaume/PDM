from parts_model import *
import numpy as np
from encoder import Encoder
from decoder import Decoder


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.n_channels = config['n_channels']
        self.n_classes = config['n_classes']
        self.no_filters = config['no_filters']
        self.interp_method = "bilinear" if config["interp_method"] is None else config["interp_method"]

        self.encoder = Encoder(config)
        self.decoder = Decoder(config, self.encoder)
        self.out_conv = OutConv(self.no_filters[0], self.n_classes)

    def forward(self, x):
        x = self.encoder(x)
        self.decoder.enc_outputs = self.encoder.enc_outputs

        x = self.decoder(x)
        logits = self.out_conv(x)
        return logits

    
class UNet_pretrained(nn.Module):
    def __init__(self, config, encoder):
        super(UNet_pretrained, self).__init__()
        self.n_channels = config['n_channels']
        self.n_classes = config['n_classes']
        self.no_filters = config['no_filters']
        self.interp_method = "bilinear" if config["interp_method"] is None else config["interp_method"]

        self.encoder = encoder.enc
        self.decoder = Decoder(config, self.encoder)
        self.out_conv = OutConv(self.no_filters[0], self.n_classes)

    def forward(self, x):
        x = self.encoder(x)
        self.decoder.enc_outputs = self.encoder.enc_outputs

        x = self.decoder(x)
        logits = self.out_conv(x)
        return logits
    
    

