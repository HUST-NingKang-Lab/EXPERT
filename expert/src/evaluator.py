import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from tqdm import trange, tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, auc
from expert.src.utils import zero_weight_unk
from pprint import pprint
from joblib import delayed
from functools import reduce


class Evaluator:

    def __init__(self, predictions_multilayer: list, actual_sources_multilayer: list,
                 num_thresholds, sample_count_threshold, par=None, nafill=0):
        self.predictions_multilayer = predictions_multilayer
        self.actual_sources_multilayer = actual_sources_multilayer
        self.n_layers = len(actual_sources_multilayer)
        labels_multilayer = [actual_sources_multilayer[layer].drop(columns=['Unknown'], errors='ignore').columns.to_series()
                             for layer in trange(self.n_layers)]
        self.labels_multilayer = [labels_multilayer[layer][actual_sources_multilayer[layer].drop(columns=['Unknown'],
                                                                                                 errors='ignore').sum() > 0]
                                  for layer in trange(self.n_layers)]
        self.num_thresholds = num_thresholds
        self.thresholds = (np.arange(num_thresholds+2) / num_thresholds).reshape(num_thresholds+2, 1) # col vector
        self.sample_count_threshold = sample_count_threshold
        # skip evaluation of un-labeled data
        self.sample_weight = [zero_weight_unk(y=actual_sources, sample_weight=np.ones(actual_sources.shape[0]))
                              for actual_sources in actual_sources_multilayer]
        self.par = par
        self.nafill = nafill
        '''self.lw = 1
        self.colors = ListedColormap(sns.color_palette("husl", 4))
        colors = self.colors.colors'''
        #self.cmap = {name: color for name, color in zip(self.score_names + ['L'], colors)}

    def eval(self):
        metrics_layers = []
        avg_metrics_layers = []
        for layer in trange(self.n_layers):
            labels = self.labels_multilayer[layer]
            predictions = self.predictions_multilayer[layer]
            actual_sources = self.actual_sources_multilayer[layer]
            sample_weight = self.sample_weight[layer]
            metrics_layer = dict(self.par(delayed(eval_single_label)(predictions[label], actual_sources[label],
                                                                     sample_weight, self.thresholds, self.nafill)
                                          for label in labels))
            metrics_layers.append(metrics_layer)
            sample_count_layer = actual_sources.drop(columns='Unknown').sum()

            # list all labels to be averaged in order to calculate metrics for a layer
            avg_labels = list(sample_count_layer[sample_count_layer > self.sample_count_threshold].index)
            # list all metrics to be averaged in order to calculate metrics for a layer
            avg_metrics = ['Acc', 'Sn', 'Sp', 'TPR', 'FPR', 'Rc', 'Pr', 'F1', 'F-max', 'ROC-AUC']
            avg_metrics_layer = pd.DataFrame(np.concatenate( [np.expand_dims(metrics_layer[label][avg_metrics].to_numpy(), 2)
                                                 for label in avg_labels], axis=2).mean(axis=2), columns=avg_metrics)
            avg_metrics_layer = avg_metrics_layer.round(4)
            avg_metrics_layers.append(avg_metrics_layer)
        all_metrics = reduce(lambda x, y: {**x, **y}, metrics_layers)
        overall_metrics = pd.concat(map(lambda label_metrics: label_metrics[1].loc[0.00, ['ROC-AUC', 'F-max']].rename(label_metrics[0]), all_metrics.items()), axis=1).T
        return metrics_layers, avg_metrics_layers, overall_metrics


def eval_single_label(predictions: pd.Series, actual_sources: pd.Series, sample_weight, thresholds, nafill):
    label = actual_sources.name
    print('Evaluating biome source:', label)

    # calculate predicted label for samples
    # samples with contribution above the threshold are considered as POSITIVE, otherwise NEGATIVE
    # This is a vectorized version using numpy broadcasting
    pred_source = (predictions.to_numpy().reshape(1, predictions.shape[0]) >= thresholds).astype(np.uint)
    pred_source = pd.DataFrame(pred_source, columns=predictions.index)
    actual_sources = actual_sources.astype(np.uint)
    metrics = pd.DataFrame()
    metrics['t'] = thresholds.flatten()
    num_thresholds = metrics['t'].shape[0] - 2
    # calculate TP, TN, FN, FP using sklearn
    conf_matrix = metrics['t'].apply(lambda T: confusion_matrix(actual_sources, pred_source.iloc[int(T * num_thresholds), :], sample_weight=sample_weight, labels=[0, 1]).ravel())
    conf_metrics = pd.DataFrame(conf_matrix.tolist(), columns=['TN', 'FP', 'FN', 'TP']).astype(np.int)
    metrics = pd.concat( (metrics, conf_metrics), axis=1).set_index('t')
    metrics['Acc'] = metrics[['TP', 'TN']].sum(axis=1) / metrics.sum(axis=1)
    metrics['Sn'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['Sp'] = metrics['TN'] / (metrics['TN'] + metrics['FP'])
    metrics['TPR'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['FPR'] = metrics['FP'] / (metrics['TN'] + metrics['FP'])
    metrics['Rc'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['Pr'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
    metrics = metrics.fillna(nafill)
    metrics['F1'] = (2 * metrics['Pr'] * metrics['Rc'] / (metrics['Pr'] + metrics['Rc']))
    idx = metrics.index
    metrics['ROC-AUC'] = ((metrics.loc[idx[:-1], 'TPR'].to_numpy() + metrics.loc[idx[1:], 'TPR'].to_numpy()) *
                          (metrics.loc[idx[:-1], 'FPR'].to_numpy() - metrics.loc[idx[1:], 'FPR'].to_numpy()) / 2).sum()
    '''metrics['PR-AUC'] = ((metrics.loc[idx[:-1], 'Pr'].to_numpy() + metrics.loc[idx[1:], 'Pr'].to_numpy()) *
                         (metrics.loc[idx[:-1], 'Rc'].to_numpy() - metrics.loc[idx[1:], 'Rc'].to_numpy()) / 2).sum()'''
    metrics['F-max'] = metrics['F1'].max()
    metrics = metrics.round(4)
    print(metrics)
    return label, metrics
