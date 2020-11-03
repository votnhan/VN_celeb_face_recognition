import torch
import models as model_md
from .classification_trainer import ClassificationTrainer
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AugClassificationTrainer(ClassificationTrainer):
    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler):
        super().__init__(config, model, loss, metrics, optimizer, lr_scheduler)
        idx_enc = config['trainer']['chosen_idx_enc']
        encoder_info = config['trainer']['encoders'][idx_enc]

        self.encoder = getattr(model_md, encoder_info['name'])(**\
                            encoder_info['args'])
        self.encoder.to(self.device)
        
        # freeze weights for encoder model !!!
        for param in self.encoder.parameters():
            param.requires_grad = False
    
        self.encoder.eval()

    def _train_epoch(self, epoch):
        self.model.train()
        self.reset_metrics_tracker()
        self.encoder.eval()
        for batch_idx, (data, target, id_img) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            embedding = self.encoder(data).detach()
            output = self.model(embedding)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update train loss, metrics
            self.train_loss.update(self.config['loss'], loss.item())
            for metric in self.metrics:
                self.train_metrics.update(metric.__name__, 
                                metric(output, target), n=output.size(0))

            if batch_idx % self.log_step == 0:
                self.log_for_step(epoch, batch_idx)

        log = self.train_loss.result()
        log.update(self.train_metrics.result())

        if self.do_val and (epoch % self.val_step == 0):
            val_log = self._validate_epoch(epoch)
            log.update(val_log)

        # step learning rate scheduler
        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            self.lr_scheduler.step(self.val_loss.avg(self.config['loss']))

        return log      

    def _validate_epoch(self, epoch, save_result=False):
        self.model.eval()
        self.val_loss.reset()
        self.val_metrics.reset()
        self.logger.info('Validation: ')
        self.encoder.eval()
        if save_result:
            result = []
        with torch.no_grad():
            for batch_idx, (data, target, id_img) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                embedding = self.encoder(data).detach()
                output = self.model(embedding)
                loss = self.criterion(output, target)               
                self.val_loss.update(self.config['loss'], loss.item())
                for metric in self.metrics:
                    self.val_metrics.update(metric.__name__, 
                                    metric(output, target), n=output.size(0))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('{}/{}'.format(batch_idx, 
                                    len(self.val_loader)))
                    self.logger.debug('{}: {}'.format(self.config['loss'], 
                                    self.val_loss.avg(self.config['loss'])))
                                        
                    self.logger.debug(self.gen_message_log_for_metrics(self.\
                        val_metrics))

                if save_result:
                    output_cls = torch.argmax(output, 1)
                    prob_pred = [output[idx][cls].exp().item() for idx, cls in \
                                    enumerate(output_cls)] 
                    result.append([id_img, target, output_cls, prob_pred])

        log = self.val_loss.result()
        log.update(self.val_metrics.result())
        val_log = {'val_{}'.format(k): v for k, v in log.items()}
        
        if save_result:
            return val_log, result
        return val_log

    