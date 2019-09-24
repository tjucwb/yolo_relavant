import numpy as np
import scipy.io

class kmeans_yolo():
    def __init__(self, n_anchors, label_txt_path, anno_mat=None):
        """
        :param n_anchors: number of anchors to be clustered

        :param label_txt_path: the summary of label path (all labels in one txt file)
        (only w and h are saved here)

        :param anno_mat: the original label path
        mat format, the satndard form of anno_mat follows
        can be none if already in standard form

        """
        self.n_anchors = n_anchors
        self.label_txt_path = label_txt_path
        self.anno_mat = anno_mat

    def convert_mat_txt(self):
        anno_mat = self.anno_mat
        label_txt_path = self.label_txt_path
        if not anno_mat:
            return np.loadtxt(label_txt_path)
        label_txt_path = self.label_txt_path
        data = scipy.io.loadmat(anno_mat)
        data = data['aim_label'][0]
        f = open(label_txt_path,'w')
        for record in data:
            bbxs = record['bbx'][0][0]
            for bb in bbxs:
                f.write(' '.join([str(int(a)) for a in bb[3:]]) + '\n')
        f.close()
        return np.loadtxt(label_txt_path)

    def get_iou(self, bbxs, clusters):
        """
        this function aims to get the iou of each bbx with each cluster
        :return: the iou between bbx and clusters
        """
        n_anchor = self.n_anchors
        n = len(bbxs)
        bbxs_w = np.repeat(bbxs[:,0], n_anchor)
        bbxs_w = np.reshape(bbxs_w,(n, n_anchor))

        bbxs_h = np.repeat(bbxs[:,1], n_anchor)
        bbxs_h = np.reshape(bbxs_h, (n, n_anchor))
        cluster_w = np.repeat(clusters[:,0], n)
        cluster_w = np.reshape(cluster_w,(n, n_anchor))
        cluster_h = np.repeat(clusters[:, 1], n)
        cluster_h = np.reshape(cluster_h, (n, n_anchor))
        cluster_area = clusters[:, 0] * clusters[:, 1]
        bbx_area = bbxs[:, 0] * bbxs[:, 1]
        cluster_area = np.tile(cluster_area, (n, 1))
        bbx_area = np.tile(bbx_area,(n_anchor, 1))
        bbx_area = bbx_area.T
        w_min = np.minimum(bbxs_w, cluster_w)
        h_min = np.minimum(bbxs_h, cluster_h)
        area_averlap = w_min * h_min
        area_averlap = np.reshape(area_averlap, (n, n_anchor))
        area_over = cluster_area + bbx_area - area_averlap

        iou = area_averlap/area_over
        return iou

    def kmeans(self, bbxs):
        """
        this function is used to update the height and width of the clusters
        :return:
        """
        # initiate the clusters
        n_anchors = self.n_anchors
        np.random.seed()
        bb_num = len(bbxs)
        bb_ind = np.random.choice(bb_num, n_anchors, replace=False)
        cluster = bbxs[bb_ind, :]
        last_max_pos = np.zeros(bb_num)

        while True:
            iou = self.get_iou(bbxs, cluster)
            current_max_pos = np.argmax(iou, axis=1)
            if (current_max_pos == last_max_pos).all():
                break
            else:
                for i in range(len(cluster)):
                    if not bbxs[current_max_pos == i].size:
                        continue
                    cluster[i] = int(np.median(bbxs[current_max_pos == i, :]))
            last_max_pos = current_max_pos

        return cluster

    def save_res(self, anchors, path):
        with open(path,'w') as f:
            for anchor in anchors:
                f.write(' '.join([str(a) for a in anchor]))


if __name__ == '__main__':
    k = kmeans_yolo(3, './label.txt', 'train_anno')
    kk = k.convert_mat_txt()
    anchors = k.kmeans(kk)
    k.save_res(anchors,'./anchor.txt')
    print(anchors)