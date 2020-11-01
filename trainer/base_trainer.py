from abc import abstractmethod
from utils import *
from logger import setup_logging
from pathlib import Path
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from numpy import inf
import logging
import torch

class BaseTrainer():
    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler):
        self.config = config
        self.model = model
        self.criterion = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.metric_names = [x.__name__ for x in self.metrics]
        self.start_epoch = 1
        self.epochs = self.config['trainer']['epochs']
        self.tracked_metric, self.mode_monitor = \
                                    self.config['trainer']['tracked_metric']
        self.early_stop = self.config['trainer']['patience']
        self.save_step = self.config['trainer']['save_period']

        # Metrics 
        # Train
        self.train_loss = MetricTracker(self.config['loss'])
        self.train_metrics = MetricTracker(*self.config['metrics'])
        # Val
        self.val_loss = MetricTracker(self.config['loss'])
        self.val_metrics = MetricTracker(*self.config['metrics'])
        # Test
        self.test_loss = MetricTracker(self.config['loss'])
        self.test_metrics = MetricTracker(*self.config['metrics'])

        # create 2 folder for logging and saving checkpoints
        save_dir =  Path(self.config['trainer']['save_dir'])
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self.save_dir = save_dir / 'models' / run_id
        self.log_dir = save_dir / 'logs' / run_id
        create_folder(self.save_dir)
        create_folder(self.log_dir)

        # logging
        self.log_step = self.config['trainer']['log_step']
        setup_logging(self.log_dir)
        self.logger = logging.getLogger('trainer')

        # device 
        if self.config['trainer']['device'] == 'GPU':
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        
        # validation when training
        self.do_val = self.config['trainer']['do_validation']
        self.val_step = self.config['trainer']['validation_step']

        # save the best checkpoint
        if self.mode_monitor == 'min':
            self.mnt_best = inf
        else:
            self.mnt_best = -inf

        # resume from checkpoint 
        cp_path = self.config['trainer']['resume_path']
        if cp_path != '':
            self.resume_checkpoint(cp_path) 
    
    
    def resume_checkpoint(self, checkpoint_path):
        cp = torch.load(checkpoint_path)
        self.logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
        self.start_epoch = cp['epoch'] + 1
        self.mnt_best = cp['monitor_best']
        self.model.load_state_dict(cp['state_dict'])
        self.optimizer.load_state_dict(cp['optimizer'])
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


    def save_checkpoint(self, epoch, save_best):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.save_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.save_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def reset_metrics_tracker(self):
        self.train_loss.reset()
        self.train_metrics.reset()
        self.val_loss.reset()
        self.val_metrics.reset()
        self.test_loss.reset()
        self.test_metrics.reset()

    def setup_loader(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, track4plot=False):
        not_improve_count = 0
        self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        if track4plot:
            self.track4plot = str(self.log_dir / 'log_loss.txt')
            headers = ['Epoch', 'Train_loss', 'Validation_loss']
            append_log_to_file(self.track4plot, headers)

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            if track4plot:
                loss_name = self.config['loss']
                lines = [epoch, result.get(loss_name), result.get('val_'+loss_name)]
                lines = [str(x) for x in lines]
                append_log_to_file(self.track4plot, lines)

            # save logged information to log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # save checkpoint with the best result on configured metric
            best = False
            tracked_metric = log.get(self.tracked_metric)
            if tracked_metric:
                improved = False
                if self.mode_monitor == 'min' and tracked_metric < self.mnt_best:
                    improved = True 

                if self.mode_monitor == 'max' and tracked_metric > self.mnt_best:
                    improved = True                 

                if improved:
                    self.mnt_best = tracked_metric
                    not_improve_count = 0
                    best=True
                else:
                    not_improve_count += 1

            if not_improve_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.early_stop))
                break

            if epoch % self.save_step == 0:
                self.save_checkpoint(epoch, save_best=best)

            if isinstance(self.lr_scheduler, MultiStepLR):
                self.lr_scheduler.step()


    def eval(self, save_result=False):
        self.model.to(self.device)
        if save_result:
            log, result = self._validate_epoch(1, save_result)
            # save prediction to csv file
            res_path = str(self.save_dir / 'result.csv')
            ids, targets, predictions = [], [], []
            for batch_pred in result:
                ids += list(batch_pred[0].cpu().numpy())
                targets += list(batch_pred[1].cpu().numpy())
                predictions += list(batch_pred[2].cpu().numpy())

            save_pandas_df(zip(targets, predictions), res_path, ids, 
                            ['Target', 'Prediction'])
            print('Saved prediction to {}.'.format(res_path))
        
        else:
            log = self._validate_epoch(1)

        # print logged information to the screen
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

    def test(self, testloader):
        pass
    
    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    @abstractmethod
    def _validate_epoch(self, epoch, save_result=False):
        raise NotImplementedError
