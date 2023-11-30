from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.datasets.target_dataset import DA
from reid import models
from reid.models import stb_net
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor, UnsupervisedTargetPreprocessor, ClassUniformlySampler
from reid.utils.logging import Logger
from reid.loss import CAPMemory
from bisect import bisect_right
from reid.utils.evaluation_metrics.retrieval import PersonReIDMAP
from reid.utils.meters import CatMeter
from reid.img_grouping import img_association
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelExchange, ChannelRandomErasing
import random

def get_data(data_dir, target, height, width, batch_size, re=0, workers=8):

    dataset = DA(data_dir, target, generate_propagate_data=True)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    ca_transformer = T.Compose([
        T.Pad(10),
        T.RandomCrop((288, 144)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelExchange(gray = 2)
    ])

    propagate_loader = DataLoader(
        UnsupervisedTargetPreprocessor(dataset.target_train_original,
                                         root=osp.join(dataset.target_images_dir, dataset.target_train_path),
                                         num_cam=dataset.target_num_cam, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    ca_propagate_loader = DataLoader(
        UnsupervisedTargetPreprocessor(dataset.target_train_original,
                                         root=osp.join(dataset.target_images_dir, dataset.target_train_path),
                                         num_cam=dataset.target_num_cam, transform=ca_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.target_images_dir, dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.target_images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, query_loader, gallery_loader, propagate_loader, ca_propagate_loader


def update_train_loader(dataset, train_samples, updated_label, height, width, batch_size, re, workers,
                        all_img_cams, sample_position=7):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(EPSILON=re)
         ])

    ca_transformer = T.Compose([
        T.Pad(10),
        T.RandomCrop((288, 144)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelExchange(gray = 2)
    ])

    # obtain global accumulated label from pseudo label and cameras
    pure_label = updated_label[updated_label>=0]
    pure_cams = all_img_cams[updated_label>=0]
    accumulate_labels = np.zeros(pure_label.shape, pure_label.dtype)
    prev_id_count = 0
    id_count_each_cam = []
    for this_cam in np.unique(pure_cams):
        percam_labels = pure_label[pure_cams == this_cam]
        unique_id = np.unique(percam_labels)
        id_count_each_cam.append(len(unique_id))
        id_dict = {ID: i for i, ID in enumerate(unique_id.tolist())}
        for i in range(len(percam_labels)):
            percam_labels[i] = id_dict[percam_labels[i]]
        accumulate_labels[pure_cams == this_cam] = percam_labels + prev_id_count
        prev_id_count += len(unique_id)
    print('  sum(id_count_each_cam)= {}'.format(sum(id_count_each_cam)))
    new_accum_labels = -1*np.ones(updated_label.shape, updated_label.dtype)
    new_accum_labels[updated_label>=0] = accumulate_labels

    # update sample list
    new_train_samples = []
    for sample in train_samples:
        lbl = updated_label[sample[3]]
        if lbl != -1:
            assert(new_accum_labels[sample[3]]>=0)
            new_sample = sample + (lbl, new_accum_labels[sample[3]])
            new_train_samples.append(new_sample)

    target_train_loader = DataLoader(
        UnsupervisedTargetPreprocessor(new_train_samples, root=osp.join(dataset.target_images_dir, dataset.target_train_path),
                                         num_cam=dataset.target_num_cam, transform=ca_transformer, has_pseudo_label=True),
        batch_size=batch_size, num_workers=workers, pin_memory=True, drop_last=True,
        sampler=ClassUniformlySampler(new_train_samples, class_position=sample_position, k=4))

    return target_train_loader, len(new_train_samples)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def test_model(model, query_loader, gallery_loader):
    model.eval()

    # meters
    query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

    # init dataset
    loaders = [query_loader, gallery_loader]

    # compute query and gallery features
    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            for data in loader:
                images = data[0]
                pids = data[2]
                cids = data[3]
                images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                features = model(images)
                # save as query features
                if loader_id == 0:
                    query_features_meter.update(features.data)
                    query_pids_meter.update(pids)
                    query_cids_meter.update(cids)
                # save as gallery features
                elif loader_id == 1:
                    gallery_features_meter.update(features.data)
                    gallery_pids_meter.update(pids)
                    gallery_cids_meter.update(cids)

    query_features = query_features_meter.get_val_numpy()
    gallery_features = gallery_features_meter.get_val_numpy()

    # compute mAP and rank@k
    result = PersonReIDMAP(
        query_features, query_cids_meter.get_val_numpy(), query_pids_meter.get_val_numpy(),
        gallery_features, gallery_cids_meter.get_val_numpy(), gallery_pids_meter.get_val_numpy(), dist='cosine')

    return result.mAP, result.CMC[0], result.CMC[4], result.CMC[9], result.CMC[19]



def main(args):
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print('log_dir= ', args.logs_dir)

    # Print logs
    print('args= ', args)
    saved_label = []
    # Create data loaders
    dataset, num_classes, query_loader, gallery_loader, propagate_loader, propagate_loader_ca = get_data(
        args.data_dir, args.target, args.height, args.width, args.batch_size, args.re, args.workers)

    # Create model
    model = stb_net.MemoryBankModel(out_dim=2048, use_bnneck=args.use_bnneck)

    # Create memory bank
    cap_memory = CAPMemory(beta=args.inv_beta, alpha=args.inv_alpha, all_img_cams=dataset.target_train_all_img_cams)

    model = model.to(device)
    cap_memory = cap_memory.to(device)
    print(args.load_ckpt)
    # Load from checkpoint
    if len(args.load_ckpt)>0:
        print('  Loading pre-trained model: {}'.format(args.load_ckpt))
        trained_dict = torch.load(args.load_ckpt)
        filtered_trained_dict = {k: v for k, v in trained_dict.items() if not k.startswith('module.classifier')}
        for k in filtered_trained_dict.keys():
            if 'embeding' in k:
                print('pretrained model has key= {}'.format(k))
        model_dict = model.state_dict()
        model_dict.update(filtered_trained_dict)
        model.load_state_dict(model_dict)

    # Evaluator
    if args.evaluate:
        print("Test:")
        eval_results = test_model(model, query_loader, gallery_loader)
        print('rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f'
             % (eval_results[1], eval_results[2], eval_results[3], eval_results[4], eval_results[0]))
        return
    
    # Optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.base_lr
        weight_decay = args.weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)

    # Trainer
    trainer = Trainer(model, cap_memory)

    # Start training
    all_len = []
    for epoch in range(args.epochs):
        lr_scheduler.step(epoch)

        # image grouping
        print('Epoch {} image grouping:'.format(epoch))
        updated_label, init_intra_id_feat = img_association(model, propagate_loader_ca, min_sample=4,
                                            eps=args.thresh, rerank=True, k1=20, k2=6, intra_id_reinitialize=True)
        updated_len = len(updated_label[updated_label >= 0])
        all_len.append(updated_len)
        
        # update train loader
        new_train_loader, loader_size = update_train_loader(dataset, dataset.target_train, updated_label, args.height, args.width,
                             args.batch_size, args.re, args.workers, dataset.target_train_all_img_cams, sample_position=5)
        num_batch = int(float(loader_size)/args.batch_size)

        # train an epoch
        trainer.train(epoch, new_train_loader, optimizer, 
                            num_batch=num_batch, all_pseudo_label=torch.from_numpy(updated_label).to(torch.device('cuda')),
                            init_intra_id_feat=init_intra_id_feat)

        # test
        if (epoch+1)%1 == 0:
            if (epoch % 5 == 0):
                torch.save(model.state_dict(), osp.join(args.logs_dir, 'final_model_epoch_'+str(epoch+1)+'.pth'))
            print('Epoch ' + str(epoch + 1)+ ' Model saved.')
            print('Test with epoch {} model:'.format(epoch))
            eval_results = test_model(model, query_loader, gallery_loader)
            print('    rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f'
                 % (eval_results[1], eval_results[2], eval_results[3], eval_results[4], eval_results[0]))

        # save final model
        if (epoch+1)%args.epochs == 0:
            torch.save(model.state_dict(), osp.join(args.logs_dir, 'final_model_epoch_'+str(epoch+1)+'.pth'))
            print('Final Model saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CA assised CAP")
    # target dataset
    parser.add_argument('--target', type=str, default='market')
    # imgs setting
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=288, help="input height, default: 288")
    parser.add_argument('--width', type=int, default=144, help="input width, default: 144")
    # random erasing
    parser.add_argument('--re', type=float, default=0.5)
    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_bnneck', action='store_true', default=True)
    parser.add_argument('--pool_type', type=str, default='avgpool')
    # optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--base_lr', type=float, default=0.00035)  # for adam
    parser.add_argument('--milestones',type=int, nargs='+', default=[20, 40])  # for adam
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only", default=False)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--print_freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--load_ckpt', type=str, default='')
    # loss learning
    parser.add_argument('--inv_alpha', type=float, default=0.2, help='update rate for the memory')
    parser.add_argument('--inv_beta', type=float, default=0.07, help='temperature for contrastive loss')
    parser.add_argument('--thresh', type=int, default=0.5, help='threshold for clustering')
    args = parser.parse_args()

    main(args)


