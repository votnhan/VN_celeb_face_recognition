import argparse
import torch.optim as otpm
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import read_json
from data_loader import VNCelebDataset
from data_loader import transforms
import losses as loss_md
import models as model_md
import trainer as trainer_md


# Setting these parameters for re-producing the result
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

def main(config):
    # Create train dataloader
    train_ds_cfg = config['train_data_loader']['args']
    train_dataset = VNCelebDataset(train_ds_cfg['data_dir'], 
                            train_ds_cfg['label_file'], 
                            transforms=transforms)
    train_loader = DataLoader(train_dataset, train_ds_cfg['batch_size'], 
                            train_ds_cfg['shuffle'], 
                            num_workers=train_ds_cfg['num_workers'])

    # Create validation dataloader
    val_ds_cfg = config['val_data_loader']['args']
    val_dataset = VNCelebDataset(val_ds_cfg['data_dir'], 
                            val_ds_cfg['label_file'], 
                            transforms=transforms)
    val_loader = DataLoader(val_dataset, val_ds_cfg['batch_size'], 
                            val_ds_cfg['shuffle'], 
                            num_workers=val_ds_cfg['num_workers'])

    # Create classification model
    model = getattr(model_md, config['model']['name'])(**config['model']['args'])

    # Create criterion (loss function)
    criterion = getattr(loss_md, config['loss'])

    # Create metrics for evaluation
    metrics = [getattr(loss_md, x) for x in config['metrics']]

    # Create optimizer
    optimizer = getattr(otpm, config['optimizer']['name'])(model.parameters(),
                        **config['optimizer']['args'])

    # Create learning rate scheduler
    lr_scheduler = getattr(otpm.lr_scheduler, config['lr_scheduler']['name'])\
                    (optimizer, **config['lr_scheduler']['args'])

    # Create train procedure: classification trainer
    trainer_cls = getattr(trainer_md, config['trainer']['name'])
    trainer = trainer_cls(config, model, criterion, metrics, 
                                        optimizer, lr_scheduler)

    trainer.setup_loader(train_loader, val_loader)
    trainer.train(config['trainer']['track4plot'])

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='VNCeleb - Face Recognition')
    args_parser.add_argument('-c', '--config', default=None, type=str, help='Path of config file')
    args_parser.add_argument('-d', '--device', default=None, type=str, help='Indices of GPUs')

    args = args_parser.parse_args()

    config = read_json(args.config)
    main(config)
