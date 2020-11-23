import torch
from .base_trainer import BaseTrainer
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ClassificationTrainer(BaseTrainer):
    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler):
        super().__init__(config, model, loss, metrics, optimizer, lr_scheduler)

    def _train_epoch(self, epoch):
        self.model.train()
        self.reset_metrics_tracker()
        for batch_idx, (data, target, id_img) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
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
        if save_result:
            result = []
        with torch.no_grad():
            for batch_idx, (data, target, id_img) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
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

    def gen_message_log_for_metrics(self, metrics_tracker):
        metric_values = [metrics_tracker.avg(x) for x in self.metric_names]
        message_metrics = ', '.join(['{}: {:.6f}'.format(x, y) \
                        for x, y in zip(self.metric_names, metric_values)])
        return message_metrics

    
    def log_for_step(self, epoch, batch_idx):
        message_loss = 'Train Epoch: {} [{}]/[{}] with {}, Loss: {:.6f}'.\
                                format(epoch, 
                                batch_idx, len(self.train_loader), 
                                self.criterion.__class__.__name__,
                                self.train_loss.avg(self.config['loss']))

        message_metrics = self.gen_message_log_for_metrics(self.train_metrics)

        self.logger.info(message_loss)
        self.logger.info(message_metrics)

    