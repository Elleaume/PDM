from parts_model import *
import numpy as np

class Decoder(nn.Module):
    def __init__(self, config, encoder):
        super(Decoder, self).__init__()
        self.n_channels = config['n_channels']
        self.n_classes = config['n_classes']
        self.no_filters = config['no_filters']
        self.n_dec_blocks = config['n_dec_blocks']
        self.encoder = encoder
        self.interp_method = "bilinear" if config["interp_method"] is None else config["interp_method"]
        self.factor = 2 if config["factor"] is None else config["factor"]
        self.enc_outputs = encoder.enc_outputs

        self.layer_list = nn.ModuleList([])
        
        self.filters_reversed = self.no_filters[::-1]
        
        for i, filters in enumerate(self.filters_reversed):
            if i + 1 > self.n_dec_blocks:
                break
            # print("Appending filter from %d to %d" % (filters,self.filters_reversed[i+1]))
            self.layer_list.add_module("up%d" % (i + 1), Up(filters, self.filters_reversed[i+1], self.interp_method))
        
    def forward(self, x):
        for i, module in enumerate (self.layer_list):
            out = module(x, self.encoder.enc_outputs[-(i+1)])
            x = out
        return x