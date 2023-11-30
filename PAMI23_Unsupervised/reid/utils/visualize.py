import numpy as np
import os
import os.path as osp
import shutil


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))


def visualize_ranked_results(distmat, dataset, save_dir='', topk=20, query_root='', gallery_root=''):
    """Visualizes ranked results.
    Supports both image-reid and video-reid.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)
    make_dirs(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            make_dirs(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src)[:9]+'.jpg')
            shutil.copy(src, dst)

    high_acc_list = []
    high_acc_thresh = 7

    for q_idx in range(num_q):
        q_infos = query[q_idx]
        qimg_path, qpid, qcamid = q_infos[0], q_infos[1], q_infos[2]
        #qimg_path, qpid, qcamid = query[q_idx]
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0])[:-4])
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path)[:-4])
        #make_dirs(qdir)
        #_cp_img_to(query_root + qimg_path, qdir, rank=0, prefix='query')
        top_hit, top_miss = 0, 0

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            g_infos = gallery[g_idx]
            gimg_path, gpid, gcamid = g_infos[0], g_infos[1], g_infos[2]
            #gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)   #original version
            invalid2 = (gpid==-1)  # added: ignore junk images
            if not (invalid or invalid2):
                if qpid != gpid:  # and rank_idx == 1:
                    top_miss += 1          
                #_cp_img_to(gallery_root + gimg_path, qdir, rank=rank_idx, prefix='gallery')
                rank_idx += 1
                if rank_idx > topk:
                    break

        if top_miss>1 and top_miss<=5:  #top_miss==1:  #top_hit < high_acc_thresh:
            high_acc_list.append(osp.basename(qimg_path)[0:7])
            # save top-ranked images for the query
            make_dirs(qdir)
            _cp_img_to(query_root + qimg_path, qdir, rank=0, prefix='query')
            rank_idx = 1
            for g_idx in indices[q_idx, :]:
                g_infos = gallery[g_idx]
                gimg_path, gpid, gcamid = g_infos[0], g_infos[1], g_infos[2]
                invalid = (qpid == gpid) & (qcamid == gcamid)   #original version
                invalid2 = (gpid==-1)  # added: ignore junk images
                if not (invalid or invalid2):
                    _cp_img_to(gallery_root + gimg_path, qdir, rank=rank_idx, prefix='gallery')
                    rank_idx += 1
                    if rank_idx > topk:
                        break

    print("Done")
    print('query images whose top-{} has mismatches are:'.format(topk))
    for elem in high_acc_list:
        print(elem)
