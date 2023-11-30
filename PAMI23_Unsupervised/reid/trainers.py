from __future__ import print_function, absolute_import
import time
import torch
from .utils.meters import AverageMeter
import torch.nn.functional as F
import numpy as np


class Trainer(object):
    def __init__(self, model, model_inv):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model_inv = model_inv

    def train(self, epoch, target_train_loader, optimizer, num_batch=100,
              all_pseudo_label='', init_intra_id_feat=''):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        # Target iter
        target_iter = iter(target_train_loader)

        # Train
        #loss_print = {}
        for batch_ind in range(num_batch):
            data_time.update(time.time() - end)
            loss_print = {}

            try:
                inputs = next(target_iter)
            except:
                target_iter = iter(target_train_loader)
                inputs = next(target_iter)

            ### Target inputs
            inputs_target = inputs[0].to(self.device)
            index_target = inputs[3].to(self.device)
            cam_target = inputs[4].to(self.device)

            # Target loss
            _, embed_feat = self.model(inputs_target)
            loss = self.model_inv.forwarding(embed_feat, index_target, cam_target, epoch=epoch, all_pseudo_label=all_pseudo_label,
                    batch_ind=batch_ind, init_intra_id_feat=init_intra_id_feat)

            loss_print['memo_loss'] = loss.item()
            losses.update(loss.item(), embed_feat.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        log = "Epoch: [{}][{}/{}], Time {:.3f} ({:.3f}), Data {:.3f} ({:.3f}), Loss {:.3f} ({:.3f})" \
                .format(epoch, num_batch, num_batch,
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        losses.val, losses.avg)

        for tag, value in loss_print.items():
            log += ", {}: {:.3f}".format(tag, value)
        print(log)






