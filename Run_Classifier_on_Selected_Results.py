import pandas as pd
import numpy as np
import xlrd
import xlwt
from time import time
from pathlib import Path
from numpy.linalg import inv
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, mutual_info_score



class KELMOR(ClassifierMixin, BaseEstimator):

    def __init__(self, C=100, method="full", S=None, eps=1e-5, kernel="rbf", gamma=0.1, degree=3, coef0=1, kernel_params=None):
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
        # print("coded_preds::",coded_preds.shape)
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


# ------------------------------------------------------------------------------
class results():
    def __init__(self):
        self.MZEList = []
        self.MAEList = []
        self.F1List = []
        self.MIList = []
        self.ALC_MZE = []
        self.ALC_MAE = []
        self.ALC_F1 = []
        self.ALC_MI = []

class stores():
    def __init__(self):
        self.MZEList_mean = []
        self.MZEList_std = []
        # -----------------
        self.MAEList_mean = []
        self.MAEList_std = []
        # -----------------
        self.F1List_mean = []
        self.F1List_std = []
        # -----------------
        self.MIList_mean = []
        self.MIList_std = []
        # -----------------
        self.ALC_MZE_mean = []
        self.ALC_MZE_std = []
        # -----------------
        self.ALC_MAE_mean = []
        self.ALC_MAE_std = []
        # -----------------
        self.ALC_F1_mean = []
        self.ALC_F1_std = []
        # -----------------
        self.ALC_MI_mean = []
        self.ALC_MI_std = []
        # -----------------
        self.ALC_MZE_list = []
        self.ALC_MAE_list = []
        self.ALC_F1_list = []
        self.ALC_MI_list = []

# --------------------------------------

def get_train_test_init_selected_ids(name,method,result_path,data_path, save_path):
    read_data_path = data_path.joinpath(name + ".csv")
    data = np.array(pd.read_csv(read_data_path, header=None))
    X = np.asarray(data[:, :-1], np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = data[:, -1]
    y -= y.min()
    read_path = str(result_path.joinpath(name + ".xls"))
    book = xlrd.open_workbook(read_path)
    RESULT = results()
    STORE = stores()

    for SN in book.sheet_names():
        S_time = time()   # -------record the time consumption---------
        table = book.sheet_by_name(SN)
        train_idx = []
        test_idx = []
        init_idx = []
        selected_idx = []
        for idx in table.col_values(0):
            if isinstance(idx,float):
                train_idx.append(int(idx))
        for idx in table.col_values(1):
            if isinstance(idx,float):
                test_idx.append(int(idx))
            else:
                break
        for idx in table.col_values(2):
            if isinstance(idx,float):
                init_idx.append(int(idx))
            else:
                break
        for idx in table.col_values(3):
            if isinstance(idx,float):
                selected_idx.append(int(idx))
            else:
                break
        # ---------------------------------
        X_pool = X[train_idx]
        y_pool = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        # ---------------------------------
        MZE_list = []
        MAE_list = []
        F1_list = []
        MI_list = []
        ALC_MZE = 0.0
        ALC_MAE = 0.0
        ALC_F1 = 0.0
        ALC_MI = 0.0
        # ----------------------------------------------------------- #
        # ---------------------Run Classifier------------------------ #
        # ----------------------------------------------------------- #
        init_len = len(init_idx)
        whole_len = len(selected_idx)
        for i in range(init_len,whole_len):
            current_ids = selected_idx[:i]
            # --------------------------------------
            model = KELMOR(C=100, kernel='rbf', gamma=0.1)
            model.fit(X=X_pool[current_ids], y=y_pool[current_ids])
            y_hat = model.predict(X=X_test)
            # --------------------------------------
            # --------------Metrics-----------------
            MZE = 1 - accuracy_score(y_hat, y_test)
            MAE = mean_absolute_error(y_hat, y_test)
            F1 = f1_score(y_pred=y_hat, y_true=y_test,average="macro")
            MI = mutual_info_score(labels_true=y_test, labels_pred=y_hat)

            # -------------------------------------
            MZE_list.append(MZE)
            MAE_list.append(MAE)
            F1_list.append(F1)
            MI_list.append(MI)
            ALC_MZE += MZE
            ALC_MAE += MAE
            ALC_F1 += F1
            ALC_MI += MI
            # -------------------------------------
        RESULT.MZEList.append(MZE_list)
        RESULT.MAEList.append(MAE_list)
        RESULT.F1List.append(F1_list)
        RESULT.MIList.append(MI_list)
        #-----------------------------
        RESULT.ALC_MZE.append(ALC_MZE)
        RESULT.ALC_MAE.append(ALC_MAE)
        RESULT.ALC_F1.append(ALC_F1)
        RESULT.ALC_MI.append(ALC_MI)


        print("==========time:{}".format(time()-S_time))

    STORE.MZEList_mean = np.mean(RESULT.MZEList, axis=0)
    STORE.MZEList_std = np.std(RESULT.MZEList, axis=0)
    STORE.MAEList_mean = np.mean(RESULT.MAEList, axis=0)
    STORE.MAEList_std = np.std(RESULT.MAEList, axis=0)
    STORE.F1List_mean = np.mean(RESULT.F1List, axis=0)
    STORE.F1List_std = np.std(RESULT.F1List, axis=0)
    STORE.MIList_mean = np.mean(RESULT.MIList, axis=0)
    STORE.MIList_std = np.std(RESULT.MIList, axis=0)
    # ------------------------------
    STORE.ALC_MZE_mean = np.mean(RESULT.ALC_MZE)
    STORE.ALC_MZE_std = np.std(RESULT.ALC_MZE)
    STORE.ALC_MAE_mean = np.mean(RESULT.ALC_MAE)
    STORE.ALC_MAE_std = np.std(RESULT.ALC_MAE)
    STORE.ALC_F1_mean = np.mean(RESULT.ALC_F1)
    STORE.ALC_F1_std = np.std(RESULT.ALC_F1)
    STORE.ALC_MI_mean = np.mean(RESULT.ALC_MI)
    STORE.ALC_MI_std = np.std(RESULT.ALC_MI)
    # ------------------------------------
    STORE.ALC_MZE_list = RESULT.ALC_MZE
    STORE.ALC_MAE_list = RESULT.ALC_MAE
    STORE.ALC_F1_list = RESULT.ALC_F1
    STORE.ALC_MI_list = RESULT.ALC_MI

    # -------------------------------------------------------
    sheet_names = ["MZE_mean", "MZE_std", "MAE_mean", "MAE_std", "F1_mean", "F1_std", "MI_mean", "MI_std", "NMI_mean", "NMI_std",
                   "ALC_MZE_list","ALC_MAE_list","ALC_F1_list","ALC_MI_list","ALC_MZE", "ALC_MAE", "ALC_F1", "ALC_MI", "ALC_NMI"]
    workbook = xlwt.Workbook()
    for sn in sheet_names:
        print("sn::",sn)
        sheet = workbook.add_sheet(sn)
        n_col = len(STORE.MZEList_mean)
        if sn == "MZE_mean":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MZEList_mean[j - 1])
        elif sn == "MZE_std":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MZEList_std[j - 1])
        elif sn == "MAE_mean":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MAEList_mean[j - 1])
        elif sn == "MAE_std":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MAEList_std[j - 1])
        elif sn == "F1_mean":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.F1List_mean[j - 1])
        elif sn == "F1_std":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.F1List_std[j - 1])

        elif sn == "MI_mean":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MIList_mean[j - 1])
        elif sn == "MI_std":
            sheet.write(0, 0, method)
            for j in range(1,n_col + 1):
                sheet.write(0,j,STORE.MIList_std[j - 1])
        # ---------------------------------------------------
        elif sn == "ALC_MZE_list":
            sheet.write(0, 0, method)
            for j in range(1,len(STORE.ALC_MZE_list) + 1):
                sheet.write(0,j,STORE.ALC_MZE_list[j - 1])
        elif sn == "ALC_MAE_list":
            sheet.write(0, 0, method)
            for j in range(1,len(STORE.ALC_MAE_list) + 1):
                sheet.write(0,j,STORE.ALC_MAE_list[j - 1])
        elif sn == "ALC_F1_list":
            sheet.write(0, 0, method)
            for j in range(1,len(STORE.ALC_F1_list) + 1):
                sheet.write(0,j,STORE.ALC_F1_list[j - 1])
        elif sn == "ALC_MI_list":
            sheet.write(0, 0, method)
            for j in range(1,len(STORE.ALC_MI_list) + 1):
                sheet.write(0,j,STORE.ALC_MI_list[j - 1])
        # -----------------
        elif sn == "ALC_MZE":
            sheet.write(0, 0, method)
            sheet.write(0, 1, STORE.ALC_MZE_mean)
            sheet.write(0, 2, STORE.ALC_MZE_std)
        elif sn == "ALC_MAE":
            sheet.write(0, 0, method)
            sheet.write(0, 1, STORE.ALC_MAE_mean)
            sheet.write(0, 2, STORE.ALC_MAE_std)
        elif sn == "ALC_F1":
            sheet.write(0, 0, method)
            sheet.write(0, 1, STORE.ALC_F1_mean)
            sheet.write(0, 2, STORE.ALC_F1_std)

        elif sn == "ALC_MI":
            sheet.write(0, 0, method)
            sheet.write(0, 1, STORE.ALC_MI_mean)
            sheet.write(0, 2, STORE.ALC_MI_std)

    save_path = str(save_path.joinpath(method))
    save_path = Path(save_path)
    save_path = str(save_path.joinpath(name + ".xls"))
    workbook.save(save_path)


if __name__ == '__main__':

    names_list = ["Newthyroid","Balance-scale","Thyroid","Machine","Cleveland","Housing","Computer","Obesity","Stock","Penbased"]
    methods_list = ["Random","USLC","USME","USMS","ALCE","FISTA","MCSVMA","McPAL","LogitA","AOCOI"]

    data_path = Path(r"D:\OCdata")
    for method in methods_list:
        print("@@@@@@@@@@@@@@@@@@@@@@{}".format(method))
        result_path = Path(r"E:\SelectedResult")
        save_path = Path(r"E:\ActiveLearningResult")
        result_path = str(result_path.joinpath(method))
        result_path = Path(result_path)
        for name in names_list:
            print("@@@@@@@@@@@@@@@@@@@@@@{}".format(name))
            get_train_test_init_selected_ids(name,method,result_path,data_path,save_path)



