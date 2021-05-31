""" Script to visualize the output of the data pipeline.
"""
import argparse
import cv2
import configs.full_cv_config as tr_test_config
from utils import augmentations as aug
from utils.data_loader import CDNet2014Loader

parser = argparse.ArgumentParser(description='BSUV-Net-2.0 pyTorch')
parser.add_argument('--inp_size', metavar='Input Size', dest='inp_size', type=int, default=224,
                    help='Size of the inputs. If equals 0, use the original sized images. Assumes square sized input')
args = parser.parse_args()

inp_size = args.inp_size
if inp_size == 0:
    inp_size = None
else:
    inp_size = (inp_size, inp_size)
dataset_tr = tr_test_config.datasets_tr[2]


crop_and_aug = [aug.RandomCrop(inp_size)]

crop_and_aug.append(aug.RandomJitteredCrop(inp_size))

crop_and_aug.append(
    [
        aug.RandomZoomCrop(inp_size),
        aug.RandomPanCrop(inp_size),
    ]
)


ill_global, std_ill_diff = (0.1, 0.04), (0.1, 0.04)
additional_augs = [[aug.AdditiveRandomIllumation(ill_global, std_ill_diff)]]


noise = 0.01
additional_augs_iom = [[aug.AdditiveRandomIllumation(ill_global, std_ill_diff)],
                       [aug.AdditiveNoise(noise)]]

iom_dataset = {
    'intermittentObjectMotion': dataset_tr['intermittentObjectMotion']
}
mask_transforms = [
    [aug.Resize((inp_size[0], inp_size[1]))],
    [aug.RandomCrop(inp_size)],
    *additional_augs_iom,
]

dataloader_mask = CDNet2014Loader(
    iom_dataset,
    empty_bg='manual',
    recent_bg=1,
    segmentation_ch=0,
    transforms=mask_transforms,
    multiplier=0,
    shuffle=True
)

additional_augs.append([aug.RandomMask(inp_size, dataloader_mask, mask_prob=0.1)])

noise = 0.01
additional_augs.append([aug.AdditiveNoise(noise)])

transforms_tr = [
    [aug.Resize((inp_size[0], inp_size[1]))],
    crop_and_aug,
    *additional_augs,
    [aug.ToTensor()],
]


dataloader_tr = CDNet2014Loader(
    dataset_tr, empty_bg='manual', recent_bg=1,
    segmentation_ch=0, transforms=transforms_tr,
)


for [img, mask] in dataloader_tr:

    img = img.numpy()[6:9].transpose((1, 2, 0))

    cv2.imshow("mask", mask.numpy()[0])
    cv2.imshow("img", img[:, :, ::-1])

    key = cv2.waitKey(0)

    if key == 27:
        exit(0)
