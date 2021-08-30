from parts_model import *


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.n_channels = config['n_channels']
        self.n_classes = config['n_classes']
        self.no_filters = config['no_filters']
        self.enc_outputs = []

        self.inc = DoubleConv(self.n_channels, 16)

        self.layer_list = nn.ModuleList([])
        for i, filters in enumerate(self.no_filters[:-1]):
            # print("Appending filter from %d to %d" % (filters,self.no_filters[i+1]))
            self.layer_list.add_module("down%d" % (i+1), Down(filters, self.no_filters[i + 1]))


    def forward(self, x):
        self.enc_outputs = []
        x = self.inc(x)

        self.enc_outputs.append(x)

        for i, module in enumerate(self.layer_list):
            out = module(x)
            if i < len(self.layer_list) - 1:
                self.enc_outputs.append(out)
            x = out

        return x
