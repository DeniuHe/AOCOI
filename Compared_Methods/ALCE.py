import xlwt
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from time import time
from sklearn.preprocessing import StandardScaler
from skactiveml.pool import CostEmbeddingAL
from skactiveml.utils import MISSING_LABEL

class ALCE():
    def __init__(self, X, y, labeled, budget):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = sorted(np.unique(self.y))
        self.nClass = len(self.labels)
        self.labeled = list(deepcopy(labeled))  # initial labeled instance set
        self.unlabeled = self.initialization()
        self.budgetLeft = deepcopy(budget)

    def initialization(self):
        unlabeled = [i for i in range(self.nSample)]
        for idx in self.labeled:
            unlabeled.remove(idx)
        return unlabeled

    def select(self):
        # ----------define the query function-----------
        qs = CostEmbeddingAL(classes=np.unique(self.y))
        # ----------reconstruct the labels--------------
        y = np.full(shape=self.y.shape, fill_value=MISSING_LABEL)
        y[self.labeled] = self.y[self.labeled]  # the labels of initial labeled instances
        # ----------Start the AL cycles-----------------
        while self.budgetLeft > 0:
            tar_idx = qs.query(X=self.X, y=y, batch_size=1)
            y[tar_idx] = self.y[tar_idx]
            self.budgetLeft -= 1
            '''
            Record the selected instances idx.
            Tis do not affect the above query selection.
            '''
            self.labeled.append(tar_idx[0])




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

            model = ALCE(X=X_train, y=y_train, labeled=labeled, budget=Budget)
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
        save_path = Path(r"E:\SelectedResult\ALCE")
        save_path = str(save_path.joinpath(name + ".xls"))
        workbook.save(save_path)

