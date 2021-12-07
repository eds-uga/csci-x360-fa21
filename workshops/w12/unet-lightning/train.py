import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(hparams):
    model = Unet(hparams)

    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        verbose=True,
    )
    # stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     mode='min',
    #     patience=5,
    #     verbose=True,
    # )
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=3,
        # callbacks=[EarlyStopping(monitor="accuracy",
        #                          mode='min', verbose=True, patience=5)]
    )

    trainer.fit(model)
    trainer.save_checkpoint("./my_checkpoint")

    # net = model.load_from_checkpoint("./my_checkpoint")
    # img = Image.open("./dataset/0c90b86742b2.png")
    # mask = predict(net, img, device=device)
    # mask_img.save("./out.png")

if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='lightning_logs')

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
