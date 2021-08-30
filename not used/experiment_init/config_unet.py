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
    loss="crossentropy",
    seed=400,
    crop_size=(192, 192),
    max_epochs=20,
    datadir="/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/BraTS2018/training/HGG/",
    n_slices=150,
    model="seg_unet",
    interp_method="bilinear",
    n_dec_blocks=5,
    factor=2,
    resize_size=(240, 240)
)
