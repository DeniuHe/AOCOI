import xlwt
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.svm import SVC


class MC_SVMA():
    def __init__(self, X, y, labeled, budget):
        self.X = X
        self.y = y.astype(np.int32)
        self.labels = np.sort(np.unique(self.y))
        self.nClass = len(self.labels)
        self.labeled = list(deepcopy(labeled))
        self.unlabeled = self.initialization()
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)
        self.tar = 0

    def initialization(self):
        unlabeled = [i for i in range(len(self.y))]
        for j in self.labeled:
            unlabeled.remove(j)
        return unlabeled

    def CBA(self):
        MCmodel = OrderedDict()
        for tar in self.labels:
            MCmodel[tar] = SVC(C=100, kernel='rbf', gamma=0.1, decision_function_shape="ovo")
        dist_to_boundary = np.zeros((len(self.unlabeled),self.nClass))
        for tar in self.labels:
            target = self.y[self.labeled]
            for t, lab in enumerate(target):
                if lab == tar:
                    target[t] = 1
                else:
                    target[t] = 0
            MCmodel[tar].fit(X=self.X[self.labeled], y=target)
            dist_to_boundary[:,tar] = MCmodel[tar].decision_function(self.X[self.unlabeled])
        Rejection_Dict = OrderedDict()
        for j, jdx in enumerate(self.unlabeled):
            if all(ele < 0 for ele in dist_to_boundary[j]):
                dist_list = deepcopy(dist_to_boundary[j])
                reject = 1 / np.linalg.norm(dist_list - np.mean(dist_list))
                Rejection_Dict[jdx] = reject
        if len(Rejection_Dict) > 0:
            tar_idx = max(Rejection_Dict,key=Rejection_Dict.get)
            return tar_idx
        else:
            return None

    def CCA(self):
        MCmodel = SVC(C=100, kernel='rbf', gamma=0.1, decision_function_shape="ovr")
        MCmodel.fit(X=self.X[self.labeled], y=self.y[self.labeled])
        DTB = MCmodel.decision_function(X=self.X[self.unlabeled])
        Compatibility = OrderedDict()
        for j, jdx in enumerate(self.unlabeled):
            tmp = np.where(DTB[j] > 0)[0]
            if len(tmp) > 2:
                Compatibility[jdx] = 1 / np.linalg.norm(DTB[j][tmp] - np.mean(DTB[j][tmp]))
        if len(Compatibility) > 0:
            tar_idx = max(Compatibility, key=Compatibility.get)
            return tar_idx
        else:
            return None

    def CNA(self):
        MCmodel = SVC(C=100, kernel='rbf', gamma=0.1, decision_function_shape="ovo")
        target = self.y[self.labeled]
        for i, lab in enumerate(target):
            if lab == self.tar:
                target[i] = 1
            else:
                target[i] = 0
        MCmodel.fit(X=self.X[self.labeled],y=target)
        dist_list = abs(MCmodel.decision_function(self.X[self.unlabeled]))
        if self.tar < self.labels[-1]:
            self.tar += 1
        else:
            self.tar = 0
        if len(dist_list) > 0:
            ord_ids = np.argsort(dist_list)
            tar_idx = self.unlabeled[ord_ids[0]]
            return tar_idx
        else:
            return None

    def select(self):
        while self.budgetLeft > 0:
            for flag in range(3):
                if self.budgetLeft <= 0:
                    break
                tar_idx = None
                if flag == 0:
                    tar_idx = self.CBA()
                elif flag == 1:
                    tar_idx = self.CCA()
                elif flag ==2:
                    tar_idx = self.CNA()
                if tar_idx == None:
                    continue
                else:
                    self.unlabeled.remove(tar_idx)
                    self.labeled.append(tar_idx)
                    self.budgetLeft -= 1



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

            model = MC_SVMA(X=X_train, y=y_train, labeled=labeled, budget=Budget)
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
        save_path = Path(r"E:\SelectedResult\MCSVMA")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)
