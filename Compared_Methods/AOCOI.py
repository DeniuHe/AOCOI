import xlwt
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y

class KELMOR(ClassifierMixin, BaseEstimator):

    def __init__(self, C=100, method="full", S=None, eps=1e-5, kernel="linear", gamma=0.1, degree=3, coef0=1, kernel_params=None):
        self.C = C
        self.kernel = kernel
        self.method = method
        self.S = S
        self.eps = eps
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X, self.y = X, y
        n, d = X.shape
        #  ---------------规范化类别标签：0,1,2,3,4,5-----------------
        self.le_ = preprocessing.LabelEncoder()
        self.le_.fit(y)
        y = self.le_.transform(y)
        #  --------------------------------------------------------
        classes = np.unique(y)
        nclasses = len(classes)

        self.M = np.array([[(i - j) ** 2 for i in range(nclasses)] for j in range(nclasses)])
        T = self.M[y, :]
        K = self._get_kernel(X)
        if self.method == "full":
            self.beta = np.linalg.inv((1 / self.C) * np.eye(n) + K).dot(T)
        else:
            raise ValueError("Invalid value for argument 'method'.")
        return self

    def predict(self, X):
        K = self._get_kernel(X, self.X)
        coded_preds = K.dot(self.beta)
        predictions = np.argmin(np.linalg.norm(coded_preds[:, None] - self.M, axis=2, ord=1), axis=1)
        predictions = self.le_.inverse_transform(predictions)
        return predictions

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {'gamma': self.gamma,
                      'degree': self.degree,
                      'coef0': self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def predict_proba(self,X):
        K = self._get_kernel(X, self.X)
        coded_tmp = K.dot(self.beta)
        predictions = np.linalg.norm(coded_tmp[:, None] - self.M, axis=2, ord=2)
        predictions = -predictions
        predictions = np.exp(predictions)
        predictions_sum = np.sum(predictions, axis=1, keepdims=True)
        proba_matrix = predictions / predictions_sum
        return proba_matrix


class AOCOI():
    def __init__(self, X, y, labeled, budget, cluster_label,cluster_center):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.labeled = list(deepcopy(labeled))
        self.gamma = 0.1
        self.model = KELMOR(C=100, kernel='rbf', gamma=self.gamma)
        self.cost_matrix = self.cal_cost_matrix()
        self.unlabeled = self.initialization()
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)
        self.cluster_label = cluster_label
        self.cluster_center = cluster_center

    def cal_cost_matrix(self):
        cost_matrix = np.zeros((len(self.labels),len(self.labels)))
        for i, y1 in enumerate(self.labels):
            for j, y2 in enumerate(self.labels):
                cost_matrix[i,j] = abs(y1-y2)
        return cost_matrix

    def initialization(self):
        unlabeled = [i for i in range(self.nSample)]
        for idx in self.labeled:
            unlabeled.remove(idx)
        self.model.fit(self.X[self.labeled], self.y[self.labeled])
        return unlabeled

    def evaluation(self):
        self.model.fit(self.X[self.labeled], self.y[self.labeled])



    def select(self):
        while self.budgetLeft > 0:
            represent_point = OrderedDict()
            n_cluster = len(self.labeled) + 1
            kmeans = KMeans(n_clusters= n_cluster)
            kmeans.fit(self.X)
            empty_cluster = OrderedDict()
            empty_center = OrderedDict()
            for lab in range(n_cluster):
                flag_set = set(self.labeled) & set(np.where(kmeans.labels_==lab)[0])
                if not flag_set:
                    empty_cluster[lab] = np.where(kmeans.labels_==lab)[0]
                    empty_center[lab] = kmeans.cluster_centers_[lab]
            # --------获取无标记簇的中心-----------
            for lab, ids in empty_cluster.items():
                min_dist = np.inf
                for idx in ids:
                    dist = np.linalg.norm(self.X[idx] - empty_center[lab])
                    if dist < min_dist:
                        min_dist = dist
                        represent_point[lab] = idx
            candidate = list(represent_point.values())

            if len(candidate) == 1:
                tar_idx = candidate[0]
                self.unlabeled.remove(tar_idx)
                self.labeled.append(tar_idx)
                self.budgetLeft -= 1
                self.evaluation()
            elif len(candidate) > 1:
                prob_matrix = self.model.predict_proba(self.X[candidate])
                MS = OrderedDict()
                for i, idx in enumerate(candidate):
                    ordjdx = np.argsort(prob_matrix[i])
                    MS[idx] = prob_matrix[i][ordjdx[-1]] - prob_matrix[i][ordjdx[-2]]

                EER = OrderedDict()
                for i, idx in enumerate(candidate):
                    # print("idx的概率估计::",prob_matrix[i])
                    tmp_model = KELMOR(C=100, kernel='rbf', gamma=0.1)
                    tmp_labeled = deepcopy(self.labeled)
                    tmp_labeled.append(idx)
                    tmp_X = self.X[tmp_labeled]
                    tmp_y = self.y[tmp_labeled]

                    Expected_Cost=0.0
                    for l, lab in enumerate(self.labels):
                        tmp_y[-1] = lab
                        tmp_model.fit(X=tmp_X, y=tmp_y)
                        tmp_prob_matrix = tmp_model.predict_proba(X=self.X[self.unlabeled])
                        tmp_prob_max = np.argmax(tmp_prob_matrix,axis=1)
                        tmp_cost_matrix = np.zeros_like(tmp_prob_matrix)
                        for t, tm in enumerate(tmp_prob_max):
                            tmp_cost_matrix[t] = self.cost_matrix[tm]
                        tmp_cost_total = tmp_cost_matrix * tmp_prob_matrix
                        tmp_cost_mean = np.mean(np.sum(tmp_cost_total,axis=1))
                        Expected_Cost += tmp_cost_mean * prob_matrix[i][l]
                    EER[idx] = Expected_Cost

                metric = OrderedDict()
                for idx in candidate:
                    metric[idx] = MS[idx] + EER[idx]

                tar_idx = min(metric, key=metric.get)
                self.unlabeled.remove(tar_idx)
                self.labeled.append(tar_idx)
                self.budgetLeft -= 1
                self.evaluation()


if __name__ == '__main__':

    names_list = ["Newthyroid","Balance-scale","Thyroid","Machine","Cleveland","Housing","Computer","Obesity","Stock","Penbased"]

    for name in names_list:
        print("########################{}".format(name))
        data_path = Path(r"D:\OCdata")
        partition_path = Path(r"E:\partition")
        """--------------read the whole data--------------------"""
        read_data_path = data_path.joinpath(name + ".csv")
        data = np.array(pd.read_csv(read_data_path, header=None))
        X = np.asarray(data[:, :-1], np.float64)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = data[:, -1]
        y -= y.min()
        nClass = len(np.unique(y))
        Budget = 10 * nClass

        """--------read the partitions--------"""
        read_partition_path = str(partition_path.joinpath(name + ".xls"))
        book_partition = xlrd.open_workbook(read_partition_path)

        workbook = xlwt.Workbook()
        count = 0
        for SN in book_partition.sheet_names():
            S_Time = time()
            train_idx = []
            test_idx = []
            labeled = []
            table_partition = book_partition.sheet_by_name(SN)
            for idx in table_partition.col_values(0):
                if isinstance(idx,float):
                    train_idx.append(int(idx))
            for idx in table_partition.col_values(1):
                if isinstance(idx,float):
                    test_idx.append(int(idx))
            for idx in table_partition.col_values(2):
                if isinstance(idx,float):
                    labeled.append(int(idx))

            X_train = X[train_idx]
            y_train = y[train_idx].astype(np.int32)
            X_test = X[test_idx]
            y_test = y[test_idx]

            model = AOCOI(X=X_train, y=y_train, labeled=labeled, budget=Budget)
            model.select()
            # SheetNames = "{}".format(count)
            # -------Store the selected results for each partition-----------
            sheet = workbook.add_sheet(SN)
            for i, idx in enumerate(train_idx):
                sheet.write(i, 0,  int(idx))
            for i, idx in enumerate(test_idx):
                sheet.write(i, 1, int(idx))
            for i, idx in enumerate(labeled):
                sheet.write(i, 2, int(idx))
            for i, idx in enumerate(model.labeled):
                sheet.write(i, 3, int(idx))
            print("SN:",SN," Time:",time()-S_Time)
        save_path = Path(r"E:\SelectedResult\AOCOI")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)

