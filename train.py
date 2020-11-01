import argparse
import torch.optim as otpm
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import read_json
from data_loader import transforms_dict
import data_loader as dataset_md
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
    # Create transforms
    transforms = transforms_dict.get(config['transforms'])

    # Create train dataloader
    train_dataset = getattr(dataset_md, config['train_dataset']['name'])(**\
                    config['train_dataset']['args'], transforms=transforms) 
                            
    train_loader_cfg = config['train_data_loader']['args']
    train_loader = DataLoader(dataset=train_dataset, **train_loader_cfg)

    # Create validation dataloader
    val_dataset = getattr(dataset_md, config['val_dataset']['name'])(**\
                    config['val_dataset']['args'], 
                    transforms=transforms_dict['default'] if transforms else None)

    val_loader_cfg = config['val_data_loader']['args']
    val_loader = DataLoader(dataset=val_dataset, **val_loader_cfg)

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
