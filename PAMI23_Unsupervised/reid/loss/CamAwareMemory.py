from __future__ import print_function, absolute_import
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math
 
torch.autograd.set_detect_anomaly(True)


class ExemplarMemory(Function):
    def __init__(self, em, alpha=0.01):
        super(ExemplarMemory, self).__init__()
        self.em = em
        self.alpha = alpha

    def forwarding(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.em.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.em)
        for x, y in zip(inputs, targets):
            self.em[y] = self.alpha * self.em[y] + (1.0 - self.alpha) * x
            self.em[y] /= self.em[y].norm()
        return grad_inputs, None


class CAPMemory(nn.Module):
    def __init__(self, beta=0.05, alpha=0.01, all_img_cams='', crosscam_epoch=5, bg_knn=50):
        super(CAPMemory, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature factor
        self.all_img_cams = torch.tensor(all_img_cams).to(torch.device('cuda'))
        self.unique_cams = torch.unique(self.all_img_cams)
        self.all_pseudo_label = ''
        self.crosscam_epoch = crosscam_epoch
        self.bg_knn = bg_knn
    
    def forwarding(self, features, targets, cams=None, epoch=None, all_pseudo_label='',
                batch_ind=-1, init_intra_id_feat=''):

        loss = torch.tensor([0.]).to(device='cuda')
        self.all_pseudo_label = all_pseudo_label
        self.init_intra_id_feat = init_intra_id_feat

        loss = self.loss_using_pseudo_percam_proxy(features, targets, cams, batch_ind, epoch)

        return loss


    def loss_using_pseudo_percam_proxy(self, features, targets, cams, batch_ind, epoch):
        if batch_ind == 0:
            # initialize proxy memory
            self.percam_memory = []
            self.memory_class_mapper = []
            self.concate_intra_class = []
            for cc in self.unique_cams:
                percam_ind = torch.nonzero(self.all_img_cams == cc).squeeze(-1)
                uniq_class = torch.unique(self.all_pseudo_label[percam_ind])
                uniq_class = uniq_class[uniq_class >= 0]
                self.concate_intra_class.append(uniq_class)
                cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
                self.memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

                if len(self.init_intra_id_feat) > 0:
                    # print('initializing ID memory from updated embedding features...')
                    proto_memory = self.init_intra_id_feat[cc]
                    proto_memory = proto_memory.to(torch.device('cuda'))
                    self.percam_memory.append(proto_memory.detach())
            self.concate_intra_class = torch.cat(self.concate_intra_class)

        if epoch >= self.crosscam_epoch:
            percam_tempV = []
            for ii in self.unique_cams:
                percam_tempV.append(self.percam_memory[ii].detach().clone())
            percam_tempV = torch.cat(percam_tempV, dim=0).to(torch.device('cuda'))

        loss = torch.tensor([0.]).to(self.device)
        for cc in torch.unique(cams):
            inds = torch.nonzero(cams == cc).squeeze(-1)
            percam_targets = self.all_pseudo_label[targets[inds]]
            percam_feat = features[inds]

            # intra-camera loss
            mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
            percam_inputs = ExemplarMemory(self.percam_memory[cc], alpha=self.alpha).forwarding(percam_feat, mapped_targets)
            percam_inputs /= self.beta  # similarity score before softmax
            loss += F.cross_entropy(percam_inputs, mapped_targets)

            # global loss
            if epoch >= self.crosscam_epoch:
                associate_loss = 0
                target_inputs = percam_feat.mm(percam_tempV.t().clone())
                temp_sims = target_inputs.detach().clone()
                target_inputs /= self.beta

                for k in range(len(percam_feat)):
                    ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_targets[k]).squeeze(-1)
                    temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                    sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn:]
                    concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                    concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
                    concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                    associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
                loss += 0.5 * associate_loss / len(percam_feat)
        return loss
