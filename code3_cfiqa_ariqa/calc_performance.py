from scipy import stats
import scipy.io as scio
import numpy as np
# import h5py
import os

from scipy.optimize import curve_fit
from scipy.optimize import leastsq

class IQAPerformance():
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.
    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, y, y_pred):
        self._y = y
        self._y_pred = y_pred
    def reset(self):
        self._y_pred = []
        self._y      = []
        self._y_std  = []

    def update(self, output):
        pred, y = output

        self._y.append(y[0].item())
        self._y_std.append(y[1].item())
        self._y_pred.append(torch.mean(pred[0]).item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        # sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))

        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        # rmse = np.sqrt(((sq - q) ** 2).mean())
        yhat,rmse = findRMSE(q,sq)
        mae = np.abs((sq - q)).mean()
        # outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()
        srocc2 = stats.spearmanr(sq, yhat)[0]
        krocc2 = stats.stats.kendalltau(sq, yhat)[0]
        plcc2 = stats.pearsonr(sq, yhat)[0]

        return srocc, krocc, plcc, rmse, mae, srocc2, krocc2, plcc2#, outlier_ratio

class IDCPerformance():
    """
    Accuracy of image distortion classification.
    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._d_pred = []
        self._d      = []

    def update(self, output):
        pred, y = output

        self._d.append(y[2].item())
        self._d_pred.append(torch.max(torch.mean(pred[1], 0), 0)[1].item())

    def compute(self):
        acc = np.mean([self._d[i] == float(self._d_pred[i]) for i in range(len(self._d))])

        return acc

def logistic(X,beta1,beta2,beta3,beta4,beta5):
    logisticPart = 0.5 - 1./(1.0+np.exp(beta2*(X-beta3)))
    return beta1*logisticPart+beta4*X+beta5

def findRMSE(x,y):
    x = x.real
    temp = stats.spearmanr(x, y)[0]
    if (temp>0):
        beta3 = np.mean(x)
        beta1 = np.abs(np.max(y) - np.min(y))
        beta4 = np.mean(y)
        beta2 = 1/np.std(x)
        beta5 = 1
    else:
        beta3 = np.mean(x)
        beta1 = -np.abs(np.max(y) - np.min(y))
        beta4 = np.mean(y)
        beta2 = 1/np.std(x)
        beta5 = 1
    try:
        popt,pconv = curve_fit(logistic,xdata=x,ydata=y,p0=(beta1,beta2,beta3,beta4,beta5),maxfev=10000000)
        #print 'popt: ',popt
        #print 'conv : ',pconv
    except:
        print('!!! error !!!')
    yhat = logistic(x,popt[0],popt[1],popt[2],popt[3],popt[4])
    rmse = np.sqrt(((y - yhat) ** 2).mean())
    return yhat,rmse