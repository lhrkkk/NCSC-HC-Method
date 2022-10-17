#coding=utf-8
#!/usr/bin/env python3
import random, argparse
from collections import Counter
try:
    import xlrd
except ImportError:
    pass
import numpy as np
from math import exp, pi, log, sqrt, sin, cos, acos
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms,safe_sparse_dot
from sklearn.manifold import TSNE
import os


#构象聚类
def _split_node(node, threshold, branching_factor):
    new_subcluster1 = _CFSubcluster()
    new_subcluster2 = _CFSubcluster()
    new_node1 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_node2 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2

    dist = euclidean_distances(
        node.centroids_, Y_norm_squared=node.squared_norm_, squared=True)
    n_clusters = dist.shape[0]

    farthest_idx = np.unravel_index(
        dist.argmax(), (n_clusters, n_clusters))
    node1_dist, node2_dist = dist[tuple([farthest_idx])]

    node1_closer = node1_dist < node2_dist
    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2

class _CFSubcluster(object):
    def __init__(self, linear_sum=None):
        if linear_sum is None:
            self.n_samples_ = 0
            self.squared_sum_ = 0.0
            self.linear_sum_ = 0
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.squared_sum_ = self.sq_norm_ = np.dot(
                self.linear_sum_, self.linear_sum_)
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.squared_sum_ += subcluster.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_
        self.sq_norm_ = np.dot(self.centroid_, self.centroid_)

    def merge_subcluster(self, nominee_cluster, threshold):
        """
        检查是否可以合并，条件符合就合并.
        """
        new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_centroid = (1 / new_n) * new_ls
        new_norm = np.dot(new_centroid, new_centroid)
        dot_product = (-2 * new_n) * new_norm
        sq_radius = (new_ss + dot_product) / new_n + new_norm
        if sq_radius <= threshold ** 2:
            (self.n_samples_, self.linear_sum_, self.squared_sum_,self.centroid_, self.sq_norm_) = new_n, new_ls, new_ss, new_centroid, new_norm
            return True
        return False
   
class _CFNode(object):
   #初始化函数
    def __init__(self, threshold, branching_factor, is_leaf, n_features):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features
        # 列表subclusters, centroids 和 squared norms一直贯穿始终
        self.subclusters_ = []
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features))
        self.init_sq_norm_ = np.zeros((branching_factor + 1))#一维列表
        self.squared_norm_ = []
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        self.init_sq_norm_[n_samples] = subcluster.sq_norm_
        # 扩容
        self.centroids_ = self.init_centroids_[:n_samples + 1, :]
        self.squared_norm_ = self.init_sq_norm_[:n_samples + 1]
    
    def update_split_subclusters(self, subcluster,new_subcluster1, new_subcluster2):
        #从一个节点去掉一个subcluster，再添加两个subcluster.
        ind = self.subclusters_.index(subcluster)#找到索引位置
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.init_sq_norm_[ind] = new_subcluster1.sq_norm_
        self.append_subcluster(new_subcluster2)

    def insert_cf_subcluster(self, subcluster):
        #插入一个新的subcluster.
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        # 首先，在树中遍历寻找与当前subcluster最近的subclusters，再将subcluster插入到此处.
        dist_matrix = np.dot(self.centroids_, subcluster.centroid_)# dot矩阵相乘
        #print(len(self.centroids_))
        dist_matrix *= -2.
        dist_matrix += self.squared_norm_
        closest_index = np.argmin(dist_matrix)
        closest_subcluster = self.subclusters_[closest_index]#距当前点最近的subclusters集
        # 如果closest_subcluster有孩子节点，递归遍历
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_cf_subcluster(subcluster)
            if not split_child:
                # 如果孩子节点没有分裂，仅需要更新closest_subcluster
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                self.init_sq_norm_[closest_index] = self.subclusters_[closest_index].sq_norm_
                return False

            # 如果发生了分割，需要重新分配孩子节点中的subclusters，并且在其父节点中添加一个subcluster.
            else:
                new_subcluster1, new_subcluster2 = _split_node(closest_subcluster.child_, self.threshold, self.branching_factor)
                self.update_split_subclusters(closest_subcluster, new_subcluster1, new_subcluster2)
                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        #没有孩子节点
        else:
            merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
            if merged:
                #更新操作
                self.init_centroids_[closest_index] =closest_subcluster.centroid_
                self.init_sq_norm_[closest_index] = closest_subcluster.sq_norm_
                return False
            # 待插入点和任何节点相距较远
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False
            # 如果没有足够的空间或者待插入点与其它点相近，则分裂操作.
            else:
                self.append_subcluster(subcluster)
                return True
           
class Birch():
   #初始化函数
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3,
                compute_labels=True):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels

    def _cal_inertia(self, X, labels):
        cntdic = {lb : 0 for lb in labels}
        for lb in labels:
            cntdic[lb] += 1
        avgdic = {lb : np.zeros(len(X[0])) for lb in labels}
        for i in range(len(X)):
            avgdic[labels[i]] += X[i]
        for lb in avgdic:
            avgdic[lb] /= cntdic[lb]
        std = 0
        for i in range(len(X)):
            std += np.linalg.norm(X[i] - avgdic[labels[i]]) ** 2
        return(std / len(X))

    def fit_try(self, X, y = None, thresholdlist = [0.5, ], branching_factor_list = [50, ], n_clusters_list = [3, ]):
        X = np.array(X)
        stdresult = np.zeros((len(thresholdlist), len(branching_factor_list), len(n_clusters_list)))
        for i, trs in enumerate(thresholdlist):
            for j, bf in enumerate(branching_factor_list):
                self.threshold = trs
                self.branching_factor = bf
                self.fit(X, global_cluster=False)
                for k, ncls in enumerate(n_clusters_list):
                    if len(self.subcluster_centers_) <= k+2:
                        continue
                    self.n_clusters = ncls
                    self._global_clustering(X)
                    std = self._cal_inertia(X, self.labels_)
                    stdresult[i][j][k] = std
                    print("cluster number: %d\tstandard deviation: %.3f" %(min(ncls, len(self.subcluster_centers_)), std))
        return(stdresult)

    def fit(self, X, y=None, global_cluster = True):
        threshold = self.threshold
        X = check_array(X, accept_sparse='csr', copy=True)
        branching_factor = self.branching_factor
        if branching_factor <= 1:
            raise ValueError("Branching_factor should be greater than one.")
        n_samples, n_features = X.shape
        #初次建立树，并且root节点是叶子.
        self.root_ = _CFNode(threshold, branching_factor, is_leaf=True,n_features=n_features)
        # 便于恢复subclusters.
        self.dummy_leaf_ = _CFNode(threshold, branching_factor,is_leaf=True, n_features=n_features)
        self.dummy_leaf_.next_leaf_ = self.root_
        self.root_.prev_leaf_ = self.dummy_leaf_
        # 未能向量化. 
        for sample in iter(X):
            subcluster = _CFSubcluster(linear_sum=sample)
            split = self.root_.insert_cf_subcluster(subcluster)
            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_, threshold, branching_factor)
                del self.root_
                self.root_ = _CFNode(threshold, branching_factor,is_leaf=False,n_features=n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids
        if global_cluster:
            self._global_clustering(X)
        else:
            self.subcluster_labels_ = np.arange(len(centroids))
            self._subcluster_norms = row_norms(
                self.subcluster_centers_, squared=True)
            if self.compute_labels:
                self.labels_ = self.predict(X)
            else:
                self.labels_ = np.zeros(len(X))
        return self

    def _get_leaves(self):
        #返回CFNode的叶子节点
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    def predict(self, X):
        reduced_distance = safe_sparse_dot(X, self.subcluster_centers_.T)
        reduced_distance *= -2
        reduced_distance += self._subcluster_norms
        return self.subcluster_labels_[np.argmin(reduced_distance, axis=1)]
   
    def _global_clustering(self, X=None):
        #对fitting之后获得的subclusters进行global_clustering
        clusterer = self.n_clusters
        centroids = self.subcluster_centers_
        compute_labels = (X is not None) and self.compute_labels

        # 预处理
        not_enough_centroids = False
        if isinstance(clusterer, int):
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
            if len(centroids) < self.n_clusters:
                not_enough_centroids = True
        elif (clusterer is not None):
            raise ValueError("n_clusters should be an instance of " "ClusterMixin or an int")

        # 避免predict环节，重复运算
        self._subcluster_norms = row_norms(
            self.subcluster_centers_, squared=True)

        if clusterer is None or not_enough_centroids:
            self.subcluster_labels_ = np.arange(len(centroids))
            if not_enough_centroids:
                pass
                #print("Number of subclusters found (%d) by Birch is less than (%d). Decrease the threshold."
                #    %(len(centroids), self.n_clusters))
        else:
            # 对所有叶子节点的subcluster进行聚类，它将subcluster的centroids作为样本，并且找到最终的centroids.
            self.subcluster_labels_ = clusterer.fit_predict(
                self.subcluster_centers_)
        if compute_labels:
            self.labels_ = self.predict(X)

class ClusterManifestor():
    def __init__(self):
        pass

    def _dihavg(self, a):
        b = np.array(a).T
        avgs = []
        for dihs in b:
            v = sum(np.array([cos(dih / 180 * pi), sin(dih / 180 * pi)]) for dih in dihs) / len(dihs)
            aavg = acos(v[0] / np.linalg.norm(v)) / pi * 180
            if v[1] < 0:
                aavg = -aavg
            avgs.append(aavg)
        return(avgs)

    def _convert_rgb(self, elist):
        clist = []
        emax, emin = max(elist), min(elist)
        mod = 255 / (emax - emin)
        for n in elist:
            n = int((n - emin) * mod)
            r, g, b = int(max(64, 255-2*n)), int(min(2*n, 511-2*n)), int(max(64, -255+2*n))
            r, g, b = hex(r).replace('0x', ''), hex(g).replace('0x', ''), hex(b).replace('0x', '')
            if len(r) == 1:
                r = '0' + r
            if len(g) == 1:
                g = '0' + g
            if len(b) == 1:
                b = '0' + b
            clist.append('#%s%s%s' %(r, g, b))
        return(clist)

    def show_data_with_energy(self, data, energy, xindex = 0, yindex = 1):
        fig = plt.figure()
        clist = self._convert_rgb(energy)
        xlist, ylist = [p[xindex] for p in data], [p[yindex] for p in data]
        plt.scatter(xlist, ylist, c = clist)
        plt.show()

    def show_clusters(self, data, uf_list = [], watershed = 100, xindex = 0, yindex = 1):
        fig = plt.figure()
        clusterdict = {}
        if uf_list == []:
            uf_list = [0 for i in range(len(data))]
        for i in range(len(uf_list)):
            if uf_list[i] not in clusterdict:
                clusterdict[uf_list[i]] = [data[i], ]
            else:
                clusterdict[uf_list[i]].append(data[i])
        clusterlist = []
        for key in clusterdict:
            clusterlist.append(clusterdict[key])

        for cluster in clusterlist:
            xlist = []
            ylist = []
            for p in cluster:
                xlist.append(p[xindex])
                ylist.append(p[yindex])
            if len(cluster) < watershed:
                plt.scatter(xlist, ylist, s = 10)
            else:
                plt.scatter(xlist, ylist, s = 5)
        plt.show()

    def scatter_cluster(self, subplot, data, uf_list = [], xindex = 0, yindex = 1):
        clusterdict = {}
        if uf_list == []:
            uf_list = [0 for a in data]
        labels = list(set(uf_list))
        clist = self._convert_rgb(labels)
        clusterdict = {lb:[] for lb in labels}
        colordict = {labels[i]:clist[i] for i in range(len(labels))}
        for i in range(len(uf_list)):
            clusterdict[uf_list[i]].append(data[i])
    
        for lb in clusterdict:
            cluster = np.array(clusterdict[lb])
            cent = np.zeros(len(cluster[0]))
            for d in cluster:
                cent += d
            cent /= len(cluster)
            xlist, ylist = [], []
            for p in clusterdict[lb]:
                xlist.append(p[xindex])
                ylist.append(p[yindex])
            subplot.scatter(xlist, ylist, c = colordict[lb], s = 5)
            subplot.scatter(cent[xindex], cent[yindex], c = colordict[lb], s = 200, marker = "+")

class AngleCluster():
    def __init__(self, label, data, names = [], energy = [], indexs = []):
        if names == []:
            names = ['' for i in range(len(data))]
        if energy == []:
            energy = np.zeros(len(data))
        if indexs == []:
            indexs = np.arange(start = 0, stop = len(data), step = 1, dtype = int)
        self.label = label
        self.data = np.array(data)
        self.n_data, self.n_dim = np.shape(self.data)
        self.names = names
        self.energy = np.array(energy)
        self.indexs = indexs
        args = self._calavg()
        self.aavg = args[1]
        self.astd = args[0]
        self.eavg = np.average(self.energy)
        args = self._get_center_elem()
        self.centerinx = args[0]
        self.centername = args[1]
        self.centerginx = args[2]

    def _calavg(self):
        dihdata = self.data.T
        dihstd = np.zeros(len(dihdata))
        dihavg = np.zeros(len(dihdata))
        for i, dihs in enumerate(dihdata):
            v = sum(np.array([cos(dih / 180 * pi), sin(dih / 180 * pi)]) for dih in dihs) / len(dihs)
            aavg = acos(v[0] / np.linalg.norm(v)) / pi * 180 * (-1 if v[1] < 0 else 1)
            avar = sum(min(abs(dih - aavg), (360 - abs(dih - aavg))) ** 2 for dih in dihs) / len(dihs)
            dihstd[i] = avar ** 0.5
            dihavg[i] = aavg
        return(dihstd, dihavg)

    def _get_center_elem(self):
        rmin = 180 * sqrt(len(self.data))
        cinx, cname, cginx = 0, '', self.indexs[0]
        for i, dihdata in enumerate(self.data):
            r = sqrt(sum(min(abs(dihdata[i] - self.aavg[i]), (360 - abs(dihdata[i] - self.aavg[i]))) ** 2 for i in range(len(dihdata))))
            if r < rmin:
                rmin, cinx, cname, cginx = r, i, self.names[i], self.indexs[i]
        return(cinx, cname, cginx)

def dimention_reduction_angle(data, sigma = 180, dimentionnumber = 0, weights = [], method = 'spectral', nmax = 0):
    if not weights:
        weights = [1 for i in range(len(data[0]))]
    if dimentionnumber == 0:
        dimentionnumber = len(data[0])
    def dw(a, b):
        ds = [min(abs(a[i] - b[i]), (360 - abs(a[i] - b[i]))) * weights[i] for i in range(len(a))]
        return(sqrt(sum(d ** 2 for d in ds)))
    m = len(data)
    n = len(data[0])
    mat = np.zeros((m, m))
    sigma = sqrt(1/12*sigma**2*n)
    sigma2 = 2 * sigma ** 2
    for i in range(0, m):
        for j in range(i+1, m):
            mat[i][j] = dw(data[i], data[j])
            mat[j][i] = mat[i][j]
        mat[i][i] = 0

    if method == 'tsne':
        try:
            tsne = TSNE(n_components=dimentionnumber, init='pca', random_state=0, square_distances = True)
        except TypeError: 
            tsne = TSNE(n_components=dimentionnumber, init='pca', random_state=0)
        tsne.metric = 'precomputed'
        tsne.init = 'random'
        newdata = tsne.fit_transform(mat)
    else:
        for i in range(0, m):
            for j in range(i, m):
                mat[i][j] = exp(-mat[i][j] / sigma2)
                mat[j][i] = mat[i][j]
        diag = [0 for i in range(m)]
        sqrtdiag = [0 for i in range(m)]
        for i in range(0, m):
            for j in range(0, m):
                diag[i] += mat[i][j]
            sqrtdiag[i] = np.sqrt(max(diag[i], 0))
        for i in range(0, m):
            for j in range(0, m):
                mat[i][j] /= -sqrtdiag[i] * sqrtdiag[j]
        for i in range(0, m):
            mat[i][i] = 1
        npargs = np.linalg.eig(mat)
        nplams, npvics = npargs[0], npargs[1].T
        #生成的向量列表是一列一个向量，需要转置
        vics = list(npvics)
        #取最小特征值的dm个向量的坐标作为新维度
        if dimentionnumber == 0:
            dimentionnumber = n
        newdata = [[vics[j][i] for j in range(0, dimentionnumber)] for i in range(0, m)]
    return(newdata)

def angle_convert(a, wetinx):
    newdata = np.zeros((len(a), len(a[0]) * 2))
    for i in range(len(a)):
        for j in range(len(a[0])):
            newdata[i][j*2] = cos(a[i][j] * pi / 180) * wetinx[j]
            newdata[i][j*2+1] = sin(a[i][j] * pi / 180) * wetinx[j]
    return(newdata)

def rearrange_clusters(data, labels, name = [], energy = []):
    if len(data) != len(labels):
        raise IndexError
    name = ['' for i in range(len(data))] if len(name) == 0 else name
    energy = np.zeros(len(data)) if len(energy) == 0 else np.array(energy)
    cnamedict = {lb:[] for lb in labels}
    cdatadict = {lb:[] for lb in labels}
    cenergydict = {lb:[] for lb in labels}
    cindexdict = {lb:[] for lb in labels}
    acdict = {}
    for i, lb in enumerate(labels):
        cnamedict[lb].append(name[i])
        cdatadict[lb].append(data[i])
        cenergydict[lb].append(energy[i])
        cindexdict[lb].append(i)
    for lb in cdatadict:
        ac = AngleCluster(lb, cdatadict[lb], cnamedict[lb], cenergydict[lb], cindexdict[lb])
        acdict[lb] = ac
    return(acdict)

def opt_result(opt_path, dihdata, energy, names, labels):
    acdict = rearrange_clusters(dihdata, labels, names, energy)
    f = open(os.getcwd() + "\\" + opt_path, "w")
    for lb in acdict:
        ac = acdict[lb]
        n, m = np.shape(ac.data)
        f.write(str(lb))
        f.write("\ncluster size,%d" %(ac.n_data))
        f.write("\ncluster center name,%s" %(ac.centername))
        f.write("\ncluster center data,")
        for i in range(m):
            f.write("%.3f "%(ac.data[ac.centerinx][i]))
        f.write(",%f" %(ac.energy[ac.centerinx]))
        f.write("\naverage conformation,")
        for i in range(m):
            f.write("%.3f "%(ac.aavg[i]))
        f.write("\nstandard division,")
        for i in range(m):
            f.write("%.3f "%(ac.astd[i]))
        f.write("\naverage energy,%f\n" %(ac.eavg))
        for i in range(ac.n_data):
            f.write(ac.names[i] + ",")
            f.write(" ".join(map(str, ac.data[i])))
            f.write("," + str(ac.energy[i]) + "\n")
    f.close()

def check_seperate(dihs, lblk = 45):
    nblk = int(360 / lblk)
    ocplst = [0 for i in range(nblk)]
    for dih in dihs:
        dih1 = dih + (360 if dih < 0 else 0)
        ocplst[min(int(dih1 // lblk), nblk-1)] = 1
    for i in range(nblk):
        if ocplst[i] == 0:
            startinx = i
            break
    else:
        return(False)
    for i in range(nblk+1):
        ii = i + startinx
        if ocplst[ii%nblk] + ocplst[(ii+1)%nblk] > 1:
            ocplst[ii%nblk] = 0
    if sum(ocplst) <= 1:
        return(False)
    return(True)

def fine_tune(dihdata0, labels, dihinxlist = [], weightlist = [], threshold = 30, iternumber = 5):
    if dihinxlist == []:
        dihinxlist = [i for i in range(len(dihdata0[0]))]
    if weightlist == []:
        weightlist = [1 for i in range(len(dihdata0[0]))]
    m = len(dihinxlist)
    newlabels = [lb for lb in labels]
    n_cls_prev = len(list(set(newlabels)))
    for iter in range(iternumber):
        newlabels = [label * 2**m for label in newlabels]
        acdict = rearrange_clusters(dihdata0, newlabels)
        for lb in acdict:
            ac = acdict[lb]
            n = ac.n_data
            dihdata = ac.data.T
            cmfcode = np.array([0 for i in range(n)])
            for j in range(len(dihinxlist)):
                dihinx = dihinxlist[j]
                dihs = dihdata[dihinx]
                cnt = [ac.data[ac.centerinx][dihinx], np.zeros(2)]
                if not check_seperate(dihs, lblk = threshold // 2):
                    continue
                for it in range(iternumber):
                    prevlbs = np.zeros(n, dtype = int)
                    cnt[1] = np.zeros(2)
                    for i, dih in enumerate(dihs):
                        if min(abs(dih - cnt[0]), (360 - abs(dih - cnt[0]))) >= threshold:
                            prevlbs[i] = 1
                        else:
                            cnt[1] += np.array([cos(dihs[0] / 180 * pi), sin(dihs[0] / 180 * pi)])
                            cnt[0] = acos(cnt[1][0] / np.linalg.norm(cnt[1])) / pi * 180 * (-1 if cnt[1][1] < 0 else 1)
                    if not np.sum(prevlbs):
                        break
                cmfcode += prevlbs * 2**j
            for i in range(n):
                newlabels[ac.indexs[i]] += cmfcode[i]
        tmp, dic = list(set(newlabels)), {}
        for i in range(len(tmp)):
            dic[tmp[i]] = i
        newlabels = [dic[lb] for lb in newlabels]
        n_cls_next = len(list(set(newlabels)))
        if n_cls_next == n_cls_prev:
            break
        n_cls_prev = n_cls_next
    return(newlabels)

def elbow_method(ydata, xdata = [], show = False):
    cnums, mdks, dk = [], [], -2**31
    data = list(map(lambda x: log(x + 1), ydata))
    if xdata == []:
        xdata = [i for i in range(len(ydata))]
    x, xplt = xdata, []
    for i in range(1, len(data) - 1):
        y1, y2 = abs(data[i] - data[i-1]), abs(data[i] - data[i+1])
        x1, x2 = abs(x[i] - x[i-1]), abs(x[i] - x[i+1])
        if(x1 * x2 * y1 * y2) == 0:
            mdks.append(-1)
            xplt.append(x[i])
            continue
        dk = (y1 * x2) / (y2 * x1)
        mdks.append(dk)
        xplt.append(x[i])
    if show:
        fig = plt.figure()
        plt.plot(xdata, data, 'o-')
        plt.show()
    s_mdks = sorted(mdks)
    s_mdks.reverse()
    s_mdks = s_mdks[0:6]
    for dk in s_mdks:
        cnums.append(xplt[mdks.index(dk)])
    return(cnums)

def cluster_main(filepath, optpath = '', maxenergy = 4, dihindex = [], weight = [], show = True):
    coldih, colenergy, colname = get_data(filepath = filepath, watershed = maxenergy, dihindex = dihindex, getname = True)
    n_cmf, n_dih = np.shape(coldih)
    wetinx = weight + [1 for i in range(len(weight), n_dih)]
    print("comformation number: %d" %(n_cmf))
    newdata = angle_convert(coldih, wetinx)
    nclslist = [i for i in range(4, 20)]
    birch = Birch(n_clusters = 2, threshold = 0.5, branching_factor=100)
    stdresult = birch.fit_try(newdata, thresholdlist=[min(0.1 * sqrt(sum(wetinx)), 2)], n_clusters_list=nclslist)
    result = stdresult[0][0]
    ncls = elbow_method(result, nclslist, show=False)[0]
    print("best cluster number: %d" %(ncls))
    birch.n_clusters = ncls
    birch.fit(newdata)
    birch.labels_ = fine_tune(coldih, birch.labels_, dihinxlist = [0,1,2,3], threshold = 90, iternumber = 5)
    print("split %d clusters into %d clusters by main dihedral angle range %d." %(ncls, len(list(set(birch.labels_))), int(180)))
    if optpath == '':
        optpath = filepath.replace(".", "_clusteropt.")
    opt_result(optpath, coldih, colenergy, colname, list(birch.labels_))
    print("cluster results output to file %s" %(optpath))
    
    if show:
        manifestor = ClusterManifestor()
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure()
        plt.axis("off")
        ax_le = fig.add_subplot(1, 2, 1, title = filepath, xticks = [], yticks = [])
        ax_tsne = fig.add_subplot(1, 2, 2, title = filepath, xticks = [], yticks = [])
        tmpinx = [i for i in range(n_cmf)]
        random.shuffle(tmpinx)
        tmpinx = tmpinx[0:300]
        data_s, labels_s = [coldih[i] for i in tmpinx], [birch.labels_[i] for i in tmpinx]
        reducteddata = dimention_reduction_angle(data_s, sigma = 30, dimentionnumber = 2, weights = wetinx, method = 'spectra')
        manifestor.scatter_cluster(ax_le, reducteddata, labels_s)
        reducteddata = dimention_reduction_angle(data_s, sigma = 30, dimentionnumber = 2, weights = wetinx, method = 'tsne')
        manifestor.scatter_cluster(ax_tsne, reducteddata, labels_s)
        plt.show()

#画态密度图
def draw_state_density0(energylist, sigma = 0.5, interval = 0.1, emax = None, error = 1, sort = True, ax = None, xr = 0, filename = ''):
    energylist = sorted(energylist)
    L = len(energylist)
    if not emax:
        emax = energylist[-1]
    emax = min(emax, energylist[-1])
    error = min(L, error)
    newenergylist = [e for e in energylist if e <= emax]
    L1 = len(newenergylist)
    const = sqrt(2 * pi) * sigma
    ecur = 0
    xdata, ydata = [], []
    Ny = 0
    if (L - L1) / const / error < 1:
        ewts = emax
    else:
        ewts = emax + sqrt(2 * sigma ** 2 * log((L - L1) / const / error))
    while ecur <= emax:
        xdata.append(ecur)
        yvalue = 0
        for e in energylist:
            if e > ewts:
                break
            yvalue += exp(-((e - ecur) / sigma) ** 2 / 2) / const / L
            Ny += 1
        ydata.append(yvalue)
        ecur += interval
    plt.title(filename)
    plt.plot(xdata, ydata)
    plt.scatter(newenergylist, [xr for i in range(len(newenergylist))])
    plt.show()

#评价方法
def gini(n, m):
    r = log(n / m, 10)
    r1, r2, r3, r4, r5 = r, r**2, r**3, r**4, r**5
    if r <= -2:
        ag = 1
    elif -2 < r <= -1:
        ag = -0.1465 * r3 - 0.9867 * r2 - 2.1965 * r1 - 0.6151
    elif -1 < r <= -0.5:
        ag = -1.3997 * r3 - 3.2343 * r2 - 2.6766 * r1 - 0.0989
    elif -0.5 < r <= 0:
        ag = 0.2041 * r2 - 0.0401 * r1 + 0.5375
    elif 0 < r <= 1.5:
        ag = 0.2487 * r5 - 0.8159 * r4 + 0.5058 * r3 + 0.4660 * r2 - 0.0699 * r1 + 0.5413
    else:
        ag = 1
    return(ag)

def GINI(data, mod = 10):
    #N, M, n, m: 构象数，总格子数，占用格子数，数据维数
    N, m = np.shape(data)
    M = (360 // mod) ** m
    blks = Counter()
    for i in range(N):
        tmp = tuple([data[i][j] // mod for j in range(m)])
        blks[tmp] += 1
    data = sorted([blks[tmp] for tmp in blks])
    n = len(data)
    vmin, vmax = data[0], data[-1]
    A0 = n * (vmax - vmin) / 2 + n * vmin
    A1 = sum(data[i] for i in range(n))
    agini = 1 if not A0 else min(A1 / A0, 1)
    sgini = gini(N, M)
    return(agini / sgini)

def NCSC(data, mod = 10):
    #N, M, n, m: 构象数，总格子数，占用格子数，数据维数
    N, m = np.shape(data)
    M = (360 // mod) ** m
    blks = Counter()
    for i in range(N):
        tmp = tuple([data[i][j] // mod for j in range(m)])
        blks[tmp] += 1
    Eblk = M * (1 - (1 - 1 / M) ** N)
    return(len(blks) / Eblk) 
	#返回的是样本分布广度，样本占用格子数 / 同数量随机均匀分布样本占用格子数

#读取csv或xls
def get_data(filepath, watershed = None, dihindex = [], getname = False):
    path = filepath
    if not os.path.exists(path):
        print("file %s does not exist!" %(filepath))
        exit()
    try:
        workbook = xlrd.open_workbook(path)
        sheet0 = workbook.sheet_by_index(0)
        colenergy = sheet0.col_values(2)
        coldih = sheet0.col_values(1)
        colname = sheet0.col_values(0)
    except Exception:
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        colenergy, coldih, colname = [], [], []
        for i in range(10):
            line = lines[i].strip("\n").split(",")
        for line in lines:
            line = line.strip("\n").split(",")
            colenergy.append(line[2])
            coldih.append(line[1])
            colname.append(line[0])

    L = len(coldih)
    for i in range(L):
        tempdihs = coldih[i].strip(" \n").split(' ')
        for j in range(len(tempdihs)):
            tempdihs[j] = float(tempdihs[j])
        coldih[i] = tempdihs
        colenergy[i] = float(colenergy[i])
    if watershed:
        for i in reversed(range(0, L)):
            if colenergy[i] > watershed:
                colenergy.pop(i)
                coldih.pop(i)
                colname.pop(i)
    if dihindex != []:
        i = 0
        while i < len(coldih):
            coldih[i] = [coldih[i][index] for index in dihindex]
            i += 1
    if getname:
        return(np.array(coldih), np.array(colenergy), colname)
    else:
        return(np.array(coldih), np.array(colenergy))

def main(args):
    if args.n <= 0:
        print("please input the number of amino acids!")
        exit()
    dihindex = [i for i in range(2 * args.n)]
    weight = [i + 1 for i in range(args.n)] + [args.n - i for i in range(args.n)]

    dihs, energys = get_data(args.i, watershed = args.e, dihindex = dihindex)
    ncsc = NCSC(dihs, mod = 60)
    gini = GINI(dihs, mod = 60)
    print("AASQ: %s\nNCMF: %d\nNCSC: %.3f\nGINI: %.3f" %(args.s, len(dihs), ncsc, gini))
    if args.stg:
        draw_state_density0(energys, emax = args.e, filename = args.i)
    if args.cluster:
        if args.sidechain:
            dihindex = []
        cluster_main(args.i, optpath = args.o, maxenergy = args.e, dihindex = dihindex, weight = weight, show = args.show)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Comformation data evaluation and cluster program.")
    parser.add_argument("-n", type = int, metavar = "N", help = "Number of amino acids", default = -1)
    parser.add_argument("-s", type = str, metavar = "S", help = "Amino acid sequence", default = "Unknown")
    parser.add_argument("-e", type = float, metavar = "E", help = "Energy truncation by kcal/mol", default = 4)
    parser.add_argument("-i", type = str, metavar = "<filename>", help = "Input dihedral angle data file", default = "")
    parser.add_argument("-o", type = str, metavar = "<filename>", help = "Cluster result output file", default = "")
    parser.add_argument("--cluster", action = "store_true", help = "Run clustering algorithm")
    parser.add_argument("--sidechain", action = "store_true", help = "Consider sidechain")
    parser.add_argument("--show", action = "store_true", help = "Visualize cluster result" )
    parser.add_argument("--stg", action = "store_true", help = "Show state density graph")
    args = parser.parse_args()
    main(args)