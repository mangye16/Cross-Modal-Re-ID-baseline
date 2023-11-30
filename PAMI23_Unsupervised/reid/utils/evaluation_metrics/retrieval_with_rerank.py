import numpy as np
from sklearn import metrics as sk_metrics
import torch
from reid.utils.rerank import re_ranking
np.set_printoptions(linewidth=2000)

class PersonReIDMAP:
    '''
    Compute Rank@k and mean Average Precision (mAP) scores
    Used for Person ReID
    Test on MarKet and Duke
    '''

    def __init__(self, query_feature, query_cam, query_label, gallery_feature, gallery_cam, gallery_label, dist, rerank=False, save_rank_result=False):
        '''
        :param query_feature: np.array, bs * feature_dim
        :param query_cam: np.array, 1d
        :param query_label: np.array, 1d
        :param gallery_feature: np.array, gallery_size * feature_dim
        :param gallery_cam: np.array, 1d
        :param gallery_label: np.array, 1d
        '''

        self.query_feature = query_feature
        self.query_cam = query_cam
        self.query_label = query_label
        self.gallery_feature = gallery_feature
        self.gallery_cam = gallery_cam
        self.gallery_label = gallery_label

        assert dist in ['cosine', 'euclidean']
        self.dist = dist

        # normalize feature for fast cosine computation
        if self.dist == 'cosine':
            self.query_feature = self.normalize(self.query_feature)
            self.gallery_feature = self.normalize(self.gallery_feature)
            distmat = np.matmul(self.query_feature, self.gallery_feature.transpose())
            #print('query-gallery cosine similarity: min= {}, max= {}'.format(distmat.min(), distmat.max()))
            distmat = 1-distmat  #distmat.max() - distmat
            if rerank:
                print('Applying person re-ranking ...')
                distmat_qq = np.matmul(self.query_feature, self.query_feature.transpose())
                distmat_gg = np.matmul(self.gallery_feature, self.gallery_feature.transpose())
                #print('query-query similarity min= {}, max= {}; gallery-gallery similarity min= {}, max= {}'.format(distmat_qq.min(), distmat_qq.max(), distmat_gg.min(), distmat_gg.max()))
                distmat_qq = distmat_qq.max() - distmat_qq
                distmat_gg = distmat_gg.max() - distmat_gg
                distmat = re_ranking(distmat, distmat_qq, distmat_gg)

            if save_rank_result:
                indices = np.argsort(distmat, axis=1)
                indices = indices[:,:100]
                print('indices shape= {}, saving distmat to result.txt'.format(indices.shape))
                np.savetxt("result.txt", indices, fmt="%04d")
                return

        if self.dist == 'euclidean':
            distmat = self.l2(self.query_feature, self.gallery_feature)
            if rerank:
                print('Applying person re-ranking ...')
                distmat_qq = self.l2(self.query_feature, self.query_feature)
                distmat_gg = self.l2(self.gallery_feature, self.gallery_feature)
                distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        APs = []
        CMC = []
        for i in range(len(query_label)):
            AP, cmc = self.evaluate(distmat[i], self.query_cam[i], self.query_label[i], self.gallery_cam, self.gallery_label)
            APs.append(AP)
            CMC.append(cmc)
            # print('{}/{}'.format(i, len(query_label)))

        self.APs = np.array(APs)
        self.mAP = np.mean(self.APs)

        min_len = 99999999
        for cmc in CMC:
            if len(cmc) < min_len:
                min_len = len(cmc)
        for i, cmc in enumerate(CMC):
            CMC[i] = cmc[0: min_len]
        self.CMC = np.mean(np.array(CMC), axis=0)

    def compute_AP(self, index, good_index):
        '''
        :param index: np.array, 1d
        :param good_index: np.array, 1d
        :return:
        '''

        num_good = len(good_index)
        hit = np.in1d(index, good_index)
        index_hit = np.argwhere(hit == True).flatten()

        if len(index_hit) == 0:
            AP = 0
            cmc = np.zeros([len(index)])
        else:
            precision = []
            for i in range(num_good):
                precision.append(float(i+1) / float((index_hit[i]+1)))
            AP = np.mean(np.array(precision))
            cmc = np.zeros([len(index)])
            cmc[index_hit[0]: ] = 1

        return AP, cmc

    def evaluate(self, per_query_dist, query_cam, query_label, gallery_cam, gallery_label):
        '''
        :param query_feature: np.array, 1d
        :param query_cam: int
        :param query_label: int
        :param gallery_feature: np.array, 2d, gallerys_size * feature_dim
        :param gallery_cam: np.array, 1d
        :param gallery_label: np.array, 1d
        :return:
        '''

        # cosine score
        #if self.dist is 'cosine':
        #    # feature has been normalize during intialization
        #    score = np.matmul(query_feature, gallery_feature.transpose())
        #    index = np.argsort(score)[::-1]
        #if self.dist is 'euclidean':
        #    #score = self.l2(query_feature.reshape([1,-1]), gallery_feature)
        index = np.argsort(per_query_dist)

        junk_index_1 = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam == gallery_cam))
        junk_index_2 = np.argwhere(gallery_label == -1)
        junk_index = np.append(junk_index_1, junk_index_2)

        good_index = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam != gallery_cam))
        index_wo_junk = self.notin1d(index, junk_index)

        return self.compute_AP(index_wo_junk, good_index)

    def in1d(self, array1, array2, invert=False):
        '''
        :param set1: np.array, 1d
        :param set2: np.array, 1d
        :return:
        '''

        mask = np.in1d(array1, array2, invert=invert)
        return array1[mask]

    def notin1d(self, array1, array2):

        return self.in1d(array1, array2, invert=True)

    def normalize(self, x):
        norm = np.tile(np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)), [1, x.shape[1]])
        return x / norm

    def cosine_dist(self, x, y):
        return sk_metrics.pairwise.cosine_distances(x, y)

    def euclidean_dist(self, x, y):
        return sk_metrics.pairwise.euclidean_distances(x, y)

    def l2(self, x, y):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)

        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        # We use clamp to keep numerical stability
        dist = torch.clamp(dist, 1e-8, np.inf)
        return dist.numpy()

