"""
Split the data into Pool Set (80%) and Testing Set (20%) with six times five-fold CV.
Seed instances are from Pool Set. with one instances randomly selected only one from each class in the Pool Set.
Store the Splitting Result.
"""
import xlwt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict


p = Path("D:\OCdata")
names_list = ["Newthyroid","Balance-scale","Thyroid","Machine","Cleveland","Housing","Computer","Obesity","Stock","Penbased"]
for name in names_list:
    path = p.joinpath(name + ".csv")
    print("########################{}".format(path))
    data = np.array(pd.read_csv(path, header=None))
    X = np.asarray(data[:, :-1], np.float64)
    y = data[:, -1]
    y -= y.min()
    nClass = len(np.unique(y))
    workbook = xlwt.Workbook()
    Rounds = 6
    count = 0
    for r in range(Rounds):
        SKF = StratifiedKFold(n_splits=5, shuffle=True)
        for train_idx, test_idx in SKF.split(X, y):
            count += 1
            X_train = X[train_idx]
            y_train = y[train_idx].astype(np.int32)
            # ------------select the seed instances--------------
            labeled = []
            label_dict = OrderedDict()
            for lab in np.unique(y_train):
                label_dict[lab] = []
            for idx in range(len(y_train)):
                label_dict[y_train[idx]].append(idx)
            for idxlist in label_dict.values():
                for jdx in np.random.choice(idxlist, size=1, replace=False):
                    labeled.append(jdx)
            # --------------------------------------------------
            # --------Store the splittings in xls file----------
            SheetNames = "{}".format(count)
            sheet = workbook.add_sheet(SheetNames)
            for i, idx in enumerate(train_idx):
                sheet.write(i, 0,  int(idx))
            for i, idx in enumerate(test_idx):
                sheet.write(i, 1, int(idx))
            for i, idx in enumerate(labeled):
                sheet.write(i, 2, int(idx))
    save_path = Path(r"E:\partition")
    save_path = str(save_path.joinpath(name + ".xls"))
    workbook.save(save_path)










