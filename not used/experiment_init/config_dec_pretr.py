from collections import OrderedDict

config = OrderedDict(
    batch_size=32,
    n_parts=4,
    temp_fac=1,
    n_channels=1,
    n_classes=4,
    no_filters=[16, 32, 64, 128, 128, 128],
    img_size=(240, 240),
    lr=0.00001,
    loss="global_dminus",
    seed=400,
    crop_size=(192, 192),
    max_epochs=20,
    datadir="/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/BraTS2018/training/HGG/",
    n_slices=150,
    n_dec_blocks=3,
    factor=2,
    interp_method="bilinear",
    model="decoder_pretrain",
    resize_size=(240, 240)
)
