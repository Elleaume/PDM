import torch
import pdb
from encoder import Encoder
from decoder import Decoder
from seg_unet import UNet
from collections import OrderedDict
from loccont_loss import LocalContrastiveLoss

config = OrderedDict(batch_size=32, n_parts=4, temp_fac=1, n_layers=5, n_dec_blocks=5, num_local_regions=9, img_size_x=192, img_size_y=192)

loss = LocalContrastiveLoss(config)

x = torch.randn((32, 192, 192))

print(loss(x))

# test_data = torch.randn((32, 16, 32, 64))
# model = UNet(16, 50)
#
# out = model(test_data)
# print("Done")
#
# pdb.set_trace()


