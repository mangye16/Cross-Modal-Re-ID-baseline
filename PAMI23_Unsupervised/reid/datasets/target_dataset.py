from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re


class DA(object):

    def __init__(self, data_dir, target, generate_propagate_data=False):

        # target image root
        self.target_images_dir = osp.join(data_dir, target)
        # training image dir
        self.target_train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
        self.target_train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.cam_dict = self.set_cam_dict()
        self.target_num_cam = self.cam_dict[target]

        self.generate_propagate_data = generate_propagate_data

        self.load()

    def set_cam_dict(self):
        cam_dict = {}
        cam_dict['market1501'] = 6
        cam_dict['dukemtmcreid'] = 8
        cam_dict['MSMT17_V1'] = 15
        cam_dict['VeRi'] = 20
        cam_dict['sysu-mm01'] = 6
        cam_dict['sysu-mm01-cam-1'] = 1
        cam_dict['sysu-mm01-cam-2'] = 2
        cam_dict['1'] =  2
        cam_dict['2'] =  2
        cam_dict['3'] =  2
        cam_dict['4'] =  2
        cam_dict['5'] =  2
        cam_dict['6'] =  2
        cam_dict['7'] =  2
        cam_dict['8'] =  2
        cam_dict['9'] =  2
        cam_dict['10'] = 2
        return cam_dict

    def preprocess(self, images_dir, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        ret = []
        if 'cuhk03' in images_dir:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.png')))
        elif 'Reg' in images_dir:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.bmp')))
        else:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            if 'cuhk03' in images_dir:
                name = osp.splitext(fname)[0]
                pid, cam = map(int, pattern.search(fname).groups())
            else:
                pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]

            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))


    def preprocess_target_train(self, images_dir, path, relabel=True):
        print('train image_dir= {}'.format(osp.join(images_dir, path)))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        all_img_prefix = {}
        ret = []
        index_to_id = {}
        all_img_cams = {}  # camera for each global index in order
        if 'cuhk03' in images_dir:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.png')))
        elif 'Reg' in images_dir:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.bmp')))
        else:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        if ('arket' in images_dir) or ('VeRi' in images_dir):
            name_segment = 4
        else:
            name_segment = 3
        
        for fpath in fpaths:
            fname = osp.basename(fpath)
            if 'cuhk03' in images_dir:
                name = osp.splitext(fname)[0]
                pid, cam = map(int, pattern.search(fname).groups())
                # bag, pid, cam, _ = map(int, name.split('_'))
                # pid += bag * 1000
            else:
                pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            cam -= 1

            split_list = fname.replace('.jpg', '').split('_')
            if name_segment == 4:
                this_prefix = split_list[0]+split_list[1]+split_list[2]+split_list[3]
            if name_segment == 3:
                this_prefix = split_list[0]+split_list[1]+split_list[2]
            if this_prefix not in all_img_prefix:
                all_img_prefix[this_prefix] = len(all_img_prefix)
            img_idx = all_img_prefix[this_prefix]  # global index

            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]

            ret.append((fname, pid, cam, img_idx))
            index_to_id[img_idx] = pid

            if this_prefix not in all_img_cams:
                all_img_cams[this_prefix] = cam

        all_img_cams = list(all_img_cams.values())
        all_img_cams = np.array(all_img_cams).astype(np.int64)
        print('  length of all_img_prefix= {}'.format(len(all_img_prefix)))
        print('  {} samples in total.'.format(len(ret)))
        print('  all cameras shape= {}, dtype= {}, unique values= {}'.format(all_img_cams.shape, all_img_cams.dtype, np.unique(all_img_cams)))

        gt_id_all_img = np.zeros(len(index_to_id.keys()))
        for index in index_to_id.keys():
            gt_id_all_img[index] = index_to_id[index]

        return ret, int(len(all_pids)), all_img_cams, len(all_img_prefix), gt_id_all_img

    def load(self):
        self.target_train, _, self.target_train_all_img_cams, self.target_train_ori_img_num, self.gt_id_all_img = \
            self.preprocess_target_train(self.target_images_dir, self.target_train_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.target_images_dir, self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.target_images_dir, self.query_path, False)
        if self.generate_propagate_data:
            self.target_train_original, _, _, _, _ = self.preprocess_target_train(self.target_images_dir, self.target_train_path)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  target train    | 'Unknown' | {:8d}"
              .format(len(self.target_train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
        if self.generate_propagate_data:
            print("  target train(ori)| 'Unknown' | {:8d}"
              .format(len(self.target_train_original)))

    def preprocess_su(self, images_dir, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        ret = []
        if 'cuhk03' in images_dir:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.png')))
        else:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        for idx, fpath in enumerate(fpaths):
            fname = osp.basename(fpath)
            if 'cuhk03' in images_dir:
                name = osp.splitext(fname)[0]
                pid, cam = map(int, pattern.search(fname).groups())
            elif 'Reg' in images_dir:
                fpaths = sorted(glob(osp.join(images_dir, path, '*.bmp')))
            else:
                pid, cam = map(int, pattern.search(fname).groups())
            pid = self.sysu_label[idx]
            if pid == -1: continue  # junk images are just ignored
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]

            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def preprocess_target_train_su(self, images_dir, path, relabel=True):
        print('train image_dir= {}'.format(osp.join(images_dir, path)))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        all_img_prefix = {}
        ret = []
        index_to_id = {}

        all_img_cams = {}  # camera for each global index in order
        if 'cuhk03' in images_dir:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.png')))
        elif 'Reg' in images_dir:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.bmp')))
        else:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        if ('arket' in images_dir) or ('VeRi' in images_dir):
            name_segment = 4
        else:
            name_segment = 3
        
        for idx, fpath in enumerate(fpaths):
            fname = osp.basename(fpath)
            if 'cuhk03' in images_dir:
                name = osp.splitext(fname)[0]
                pid, cam = map(int, pattern.search(fname).groups())
                # bag, pid, cam, _ = map(int, name.split('_'))
                # pid += bag * 1000
            else:
                pid, cam = map(int, pattern.search(fname).groups())
            pid = self.sysu_label[idx]
            if pid == -1: continue  # junk images are just ignored
            cam -= 1

            split_list = fname.replace('.jpg', '').split('_')
            if name_segment == 4:
                this_prefix = split_list[0]+split_list[1]+split_list[2]+split_list[3]
            if name_segment == 3:
                this_prefix = split_list[0]+split_list[1]+split_list[2]
            if this_prefix not in all_img_prefix:
                all_img_prefix[this_prefix] = len(all_img_prefix)
            img_idx = all_img_prefix[this_prefix]  # global index

            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]

            ret.append((fname, pid, cam, img_idx))
            index_to_id[img_idx] = pid

            if this_prefix not in all_img_cams:
                all_img_cams[this_prefix] = cam

        all_img_cams = list(all_img_cams.values())
        all_img_cams = np.array(all_img_cams).astype(np.int64)
        print('  length of all_img_prefix= {}'.format(len(all_img_prefix)))
        print('  {} samples in total.'.format(len(ret)))
        print('  all cameras shape= {}, dtype= {}, unique values= {}'.format(all_img_cams.shape, all_img_cams.dtype, np.unique(all_img_cams)))

        gt_id_all_img = np.zeros(len(index_to_id.keys()))
        for index in index_to_id.keys():
            gt_id_all_img[index] = index_to_id[index]

        return ret, int(len(all_pids)), all_img_cams, len(all_img_prefix), gt_id_all_img