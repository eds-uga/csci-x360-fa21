import os
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from Unet import Unet
from dataset import DirDataset


def predict(net, img, device='cpu', threshold=0.5):
    ds = DirDataset('', '')
    _img = torch.from_numpy(ds.preprocess(img))

    _img = _img.unsqueeze(0)
    _img = _img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        o = net(_img)

        if net.n_classes > 1:
            pass
        else:
            probs = torch.sigmoid(o)
        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img.size[1]),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()

    return mask > threshold


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Unet.load_from_checkpoint(checkpoint_path="./my_checkpoint",hparams=None)
    net.freeze()
    net.to(device)

    for fn in tqdm(os.listdir(hparams.img_dir)):
        fp = os.path.join(hparams.img_dir, fn)

        img = Image.open(fp)
        mask = predict(net, img, device=device)

        mask_img = mask_to_image(mask)
        mask_img.save(os.path.join(hparams.out_dir, fn))


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--checkpoint', required=True)
    parent_parser.add_argument('--img_dir', required=True)
    parent_parser.add_argument('--out_dir', required=True)

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
