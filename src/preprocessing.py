import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

ia.seed(2018)

def _standardize(img):
    return (img - img.map(np.mean)) / img.map(np.std)

affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Noop(),
                ]),
    iaa.Affine(rotate=(-5, 5), mode='reflect'),
    iaa.Crop(px=(0, 10)),
], random_order=True)

intensity_seq = iaa.Sequential([
    #iaa.Invert(0.3),
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.Add((-10, 10)),
                iaa.AddElementwise((-10, 10)),
                iaa.Multiply((0.95, 1.05)),
                iaa.MultiplyElementwise((0.95, 1.05)),
            ]),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.AverageBlur(k=(2, 5)),
            #iaa.MedianBlur(k=(3, 5))
        ])
    ])
], random_order=False)

tta_intensity_seq = iaa.Sequential([
    iaa.Noop()
], random_order=False)

def compute_random_pad(limit=(-4,4)):
    dy  = IMG_TAR_SIZE - IMG_ORI_SIZE*SCALE
    dy0 = dy//2 + np.random.randint(limit[0],limit[1]) # np.random.choice(dy)
    dy1 = dy - dy0
    dx0 = dy//2 + np.random.randint(limit[0],limit[1]) # np.random.choice(dy)
    dx1 = dy - dx0
    return dy0, dx0, dy1, dx1

def resize_pad_seq(pad_size):
    dy0, dx0, dy1, dx1 = compute_random_pad()
    seq = iaa.Sequential([
        affine_seq,
        iaa.Scale({'height': IMG_ORI_SIZE*SCALE, 'width': IMG_ORI_SIZE*SCALE}),
        iaa.Pad(px=(dy0, dx0, dy1, dx1), pad_mode='edge', keep_size=False),
    ], random_order=False)
    return seq

def resize_pad_seq_eval(pad_size):
    seq = iaa.Sequential([
        iaa.Scale({'height': IMG_ORI_SIZE*SCALE, 'width': IMG_ORI_SIZE*SCALE}),
        iaa.Pad(px=(pad_size, pad_size, pad_size+1, pad_size+1), pad_mode='edge', keep_size=False),
    ], random_order=False)
    return seq

def resize_seq():
    seq = iaa.Sequential([
        affine_seq,
        iaa.Scale({'height': IMG_TAR_SIZE, 'width': IMG_TAR_SIZE})
    ], random_order=False)
    return seq

def resize_seq_eval():
    seq = iaa.Sequential([
        iaa.Scale({'height': IMG_TAR_SIZE, 'width': IMG_TAR_SIZE})
    ], random_order=False)
    return seq