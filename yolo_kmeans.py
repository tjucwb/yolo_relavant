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
        self.class_num = 2

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

        _, txt_name, extention = label_txt_path.split(".")
        class_path = '.'+txt_name + '_with_class.' + extention
        f1 = open(class_path,'w')
        for record in data:
            bbxs = record['bbx'][0][0]
            for bb in bbxs:
                f1.write(' '.join([str(int(a)) for a in bb[[0,3,4]]]) + '\n')
        f1.close()

        return np.loadtxt(label_txt_path), np.loadtxt(class_path)

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

        #cluster_w = np.repeat(clusters[:,0], n)
        #cluster_w = np.reshape(cluster_w,(n, n_anchor))
        cluster_w = np.tile(clusters[:,0],(n,1))

        #cluster_h = np.repeat(clusters[:, 1], n)
        #cluster_h = np.reshape(cluster_h, (n, n_anchor))
        cluster_h = np.tile(clusters[:,1],(n,1))

        cluster_area = np.multiply(clusters[:, 0], clusters[:, 1])
        bbx_area = np.multiply(bbxs[:, 0], bbxs[:, 1])
        cluster_area = np.tile(cluster_area, (n, 1))

        bbx_area = np.tile(bbx_area,(n_anchor, 1))
        bbx_area = bbx_area.T

        w_min = np.minimum(bbxs_w, cluster_w)
        h_min = np.minimum(bbxs_h, cluster_h)
        #area_averlap = w_min * h_min
        area_averlap = np.multiply(w_min, h_min)

        #area_averlap = np.reshape(area_averlap, (n, n_anchor))

        area_over = cluster_area + bbx_area - area_averlap
        iou = area_averlap/area_over
        #print(iou)
        return iou


    def get_miou(self,bbxs,clusters):
        iou = self.get_iou(bbxs,clusters)
        print(np.sum(iou>1))
        iou = np.max(iou,axis=1)

        return np.mean(iou)

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
                    print(bbxs[current_max_pos == i,:])
                    cluster[i,:] = (np.median(bbxs[current_max_pos == i,:],axis=0))
            last_max_pos = current_max_pos

        return cluster

    def average_bbx(self,bbx_with_class):
        cn = self.class_num
        anchor = []
        class_list = bbx_with_class[:,0]
        for cl in range(cn):
            bb1 = None
            bb2 = None
            cl_data = bbx_with_class[class_list == cl, :]
            if (cl_data[:, 1] > cl_data[:, 2]).any():
                bb1 = np.median(cl_data[cl_data[:,1] > cl_data[:,2], 1:],axis=0)
            if (cl_data[:, 2] > cl_data[:, 1]).any():
                bb2 = np.median(cl_data[cl_data[:,2] > cl_data[:,1], 1:],axis=0)
            anchor.append(bb1)
            anchor.append(bb2)
        return anchor


    def save_res(self, anchors,miou, path):
        with open(path,'w') as f:
            for anchor in anchors:
                f.write(' '.join([str(a) for a in anchor])+'\n')
            f.write('miou is ' + str(miou))


if __name__ == '__main__':
    k = kmeans_yolo(6, './label.txt', 'train_anno')
    no_class, with_class = k.convert_mat_txt()
    anchors = k.kmeans(no_class)
    anchors2 = k.average_bbx(with_class)
    print(anchors2)
    miou = k.get_miou(no_class, anchors)
    k.save_res(anchors,miou,'./anchor.txt')
