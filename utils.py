#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Oct 2018 @author: Rui ZHAO
The following code builds some useful classes and tools.

"""

import os.path as op
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import math
import seaborn as sns
import pandas as pd
import openpyxl

class Data(object):

    def __init__(self, filepath, snr=None):
        self.name = op.splitext(op.split(filepath)[1])[0]
        if snr: self.name = self.name+'_noise'
        self.data = scio.loadmat(filepath)  # load data
        self.img = np.array(self.data['X'], dtype=np.float64)  # image
        self.tgt = np.array(self.data['d'], dtype=np.float64)  # target
        self.grt = np.array(self.data['groundtruth'], dtype=np.float64)  # groundtruth
        if snr:
            self.add_noise(snr)  # add noise

    def add_noise(self, snr):
        for i in range(self.img.shape[1]):   # add noise
            self.img[:, i] += wgn(self.img[:, i], snr)


class Detector(object):

    def __init__(self):
        self.data = []

    def load_data(self, img_data):
        self.data = img_data
        self.name = img_data.name
        self.img = img_data.img
        self.tgt = img_data.tgt
        self.grt = img_data.grt

    def show(self, results, names):
        imgshow = [self.img[1,].reshape(self.grt.shape, order='F'), self.grt]
        nameshow = ['image(first band)', 'groundtruth'] + names

        imgshow.append(results[0]['SAM'].reshape(self.grt.shape,order='F'))
        # print(results[0]['CEM'].shape)
        # print(self.grt.reshape(-1, 1, order='F').shape)
        # for item in results:
        #     print(item.shape)
        #     imgshow.append(item.reshape(self.grt.shape, order='F'))
        k = math.ceil(len(imgshow) / 3) * 100 + 31
        for i in range(len(imgshow)):  # show image
            plt.subplot(k + i)
            plt.axis('off')
            plt.imshow(imgshow[i], cmap='gray')
            plt.title(nameshow[i])
        # plot_ROC(self.grt.reshape(-1, 1, order='F'), results, names)  # plot ROC curve
        # plot_ROC(self.grt.reshape(-1, 1, order='F'), [results[0]['SAM']], names)

        calculate_roc_auc(np.concatenate(self.grt.reshape(-1, 1, order='F')), np.concatenate(results[0]['ACE']))
        grt_out=np.concatenate(self.grt.reshape(-1, 1, order='F'))
        pro_out=np.concatenate(results[0]['ACE'])

        df = pd.DataFrame({
            'Column1': grt_out,
            'Column2': pro_out
        })

        # 导出到Excel文件
        df.to_excel('output.xlsx', index=False)

        a=results[0]['ACE']
        b=self.grt.reshape(-1, 1, order='F')
        a_b0 = a[b == 0]
        a_b1 = a[b == 1]
        # b为0时的a的箱型图
        plt.boxplot([a_b0], labels=['background'])
        plt.boxplot([a_b1], positions=[2], labels=['target'])

        # 显示图形
        plt.show()
        return 0


def wgn(x, snr):
    # add white Gaussian noise to x with specific SNR
    snr = 10**(snr/10)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def dual_sigmoid(x):
    x = np.array(x)
    weights = 1.0 / (1.0 + np.exp(-x))
    return weights



def calculate_roc_auc(y_true, y_pred_proba):
    y_pred_proba = (y_pred_proba - np.min(y_pred_proba))/(np.max(y_pred_proba)- np.min(y_pred_proba))
    # 按预测概率排序索引
    sorted_indices = np.argsort(y_pred_proba)[::-1]

    y_true_sorted = y_true[sorted_indices]
    y_pred_proba_sorted = y_pred_proba[sorted_indices]

    # 初始化TPR和FPR列表
    tpr = []  # True Positive Rate
    fpr = []  # False Positive Rate

    # 计算TPR和FPR
    num_positives = np.sum(y_true)
    num_negatives = len(y_true) - num_positives
    tp = fp = 0

    for pred_prob, true_label in zip(y_pred_proba_sorted, y_true_sorted):
        if true_label:
            tp += 1
        else:
            fp += 1

        tpr.append(tp / num_positives)
        fpr.append(fp / num_negatives)

    # 计算AUC
    auc = np.trapz(tpr, fpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    return 0





def plot_ROC(test_labels, resultall, name):
    plt.subplots(num='ROC curve', figsize = [10,7])

    for i in range(len(resultall)):
        fpr, tpr, thresholds = metrics.roc_curve(
         test_labels, resultall[i], pos_label=1)
        auc = "%.5f" % metrics.auc(fpr, tpr)
        print('%s_AUC: %s'%(name[i],auc))
        plt.plot(fpr, tpr, label=name[i])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', facecolor='none', edgecolor='none')
    plt.title('ROC Curve')
    plt.show()


    return 0 
