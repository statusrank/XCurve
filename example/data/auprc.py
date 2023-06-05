import os
import numpy as np
import torch
import torch.nn.functional as F
import time
# from tqdm import tqdm

from XCurve.AUPRC.base_container import BaseContainer
from XCurve.AUPRC.utils.summary import TensorboardSummary
from XCurve.AUPRC.utils.logger import logger, set_logger_path
from XCurve.AUPRC.utils.utils import Saver


class Trainer(BaseContainer):
    def __init__(self):
        super().__init__()
        now_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
        logger_path = os.path.join(
            self.args.training.save_dir,
            self.args.dataset.dataset_train,
            self.args.models.model,
            self.args.training.experiment_id,
            '%s.log' % now_time
        )
        set_logger_path(logger_path)
        logger.info(self.args)

        # Define Saver
        self.saver = Saver(self.args)

        # Define Tensorboard Summary
        self.summary = TensorboardSummary()
        self.writer = self.summary.create_summary(self.saver.experiment_dir, self.args.models)

        self.init_training_container()

        # show parameters to be trained
        logger.debug('\nTraining params:')
        for p in self.model.named_parameters():
            if p[1].requires_grad:
                logger.debug(p[0])
        logger.debug('\n')

        # Clear start epoch if fine-tuning
        logger.info('Starting iteration: %d' % self.start_it)
        logger.info('Total iterationes: %d' % self.args.training.max_iter)

    # main function for training
    def training(self):
        self.model.train()

        logger.info('\nTraining')

        max_iter = self.args.training.max_iter
        it = self.start_it

        epoch = 0
        while it <= max_iter:
            self.train_loader.dataset.reset()
            for samples in self.train_loader:
                samples = to_cuda(samples)
    
                outputs = self.model(samples, mode='train')
                outputs['iter'] = it
                losses = self.criterion(outputs)
                loss = losses['loss']
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log training loss
                if it % self.args.training.log_interval == 0:
                    cur_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    cur_fc_lr = self.optimizer.state_dict()['param_groups'][1]['lr']
                    logger.info('\n===> Iteration  %d/%d' % (it, max_iter))
                    loss_log_str = '==>lr:  %.6f|%.6f    total loss: %.4f'%(cur_lr, cur_fc_lr, loss.item())
                    for loss_name in losses.keys():
                        if loss_name != 'loss':
                            loss_log_str += '    %s: %.4f'%(loss_name, losses[loss_name])
                            self.writer.add_scalar('train/%s_iter'%loss_name, losses[loss_name], it)
                    logger.info(loss_log_str)
                    self.writer.add_scalar('train/total_loss_iter', loss.item(), it)

                # adjust learning rate
                lr_decay_iter = self.args.training.optimizer.get('lr_decay_iter', [])
                for i in range(len(lr_decay_iter)):
                    if it == lr_decay_iter[i]:
                        lr = self.args.training.optimizer.lr * (self.args.training.optimizer.lr_decay ** (i+1))
                        logger.info('\nReduce lr to %.6f\n'%(lr))
                        self.optimizer.param_groups[0]['lr'] = lr
                        self.optimizer.param_groups[1]['lr'] = lr * self.args.training.optimizer.lr_fc_mul
                        break

                # save model and optimizer
                if (it > 0 and it % self.args.training.save_iter == 0) or it == max_iter:
                    logger.info('\nSaving checkpoint ......')
                    self.saver.save_checkpoint({
                        'best': self.best,
                        'start_it': it,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, filename='ckp_%06d.pth.tar'%it)
                    logger.info('Done.')

                # validation
                if not self.args.training.get('val_by_epoch', False):
                    val_iter = self.args.training.get('val_iter', -1)
                    if val_iter > 0 and it > 0 and it % val_iter == 0:
                        self.validation(it, 'val')

                it += 1
                if it > max_iter:
                    break

            epoch += 1
            # validation
            if self.args.training.get('val_by_epoch', False):
                self.validation(it, 'val', epoch=epoch)

    # main function for validation
    def validation(self, it, split, epoch=-1):
        # return 
        logger.info('\nEvaluating %s...'%split)
        self.evaluator.reset()
        self.model.eval()

        data_loader = self.val_loader if split == 'val' else self.test_loader
        for samples in data_loader:
            samples = to_cuda(samples)

            with torch.no_grad():
                outputs = self.model(samples)
                self.evaluator.add_batch(outputs['feat'], outputs['target'])

        results = self.evaluator.run()[0]
        mAP = results['mAP']
        recal_at_1 = results['recall@1']
        if epoch >= 0:
            self.writer.add_scalar('%s/mAP_epoch'%split, mAP, epoch)
            self.writer.add_scalar('%s/rec_1_epoch'%split, recal_at_1, epoch)
        else:
            self.writer.add_scalar('%s/mAP'%split, mAP, it)
            self.writer.add_scalar('%s/rec_1'%split, recal_at_1, it)

        if split == 'val':
            logger.info('=====>[Iteration: %d    %s/mAP=%.4f    previous best=%.4f'%(it, split, mAP, self.best))
            logger.info('=====>[Iteration: %d    %s/rec@1=%.4f'%(it, split, recal_at_1))
        else:
            logger.info('=====>[Iteration: %d    %s/mAP=%.4f'%(it, split, mAP))

        if split == 'val' and mAP > self.best:
            self.best = mAP
            logger.info('\nSaving checkpoint ......')
            self.saver.save_checkpoint({
                'best': self.best,
                'start_it': it,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, filename='best.pth.tar')

        self.model.train()

def to_cuda(sample):
    if isinstance(sample, list):
        return [to_cuda(i) for i in sample]
    elif isinstance(sample, dict):
        for key in sample.keys():
            sample[key] = to_cuda(sample[key])
        return sample
    elif isinstance(sample, torch.Tensor):
        return sample.cuda()
    else:
        return sample

def main():
    trainer = Trainer()
    trainer.training()
    trainer.writer.close()

if __name__ == "__main__":
    main()
