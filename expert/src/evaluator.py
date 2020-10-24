import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from tqdm import trange, tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, auc
from expert.src.utils import zero_weight_unk
from pprint import pprint

# sample weight
class Evaluator:

    def __init__(self, predictions_multilayer: list, actual_sources_multilayer: list, sample_count_threshold=100):
        self.predictions_multilayer = predictions_multilayer
        self.actual_sources_multilayer = actual_sources_multilayer
        self.n_layers = len(actual_sources_multilayer)
        self.labels_multilayer = [actual_sources_multilayer[layer].drop(columns=['Unknown']).columns.to_series()
                                  for layer in trange(self.n_layers)]
        self.thresholds = (np.arange(202) / 200).reshape(202, 1) # col vector
        self.sample_count_threshold = sample_count_threshold
        self.sample_weight = [zero_weight_unk(y=actual_sources, sample_weight=np.ones(actual_sources.shape[0]))
                              for actual_sources in actual_sources_multilayer]
        self.lw = 1
        self.colors = ListedColormap(sns.color_palette("husl", 4))
        colors = self.colors.colors
        #self.cmap = {name: color for name, color in zip(self.score_names + ['L'], colors)}

    def eval(self):
        metrics_layers = []
        avg_metrics_layers = []
        for layer in trange(self.n_layers):
            labels = self.labels_multilayer[layer]
            predictions = self.predictions_multilayer[layer]
            actual_sources = self.actual_sources_multilayer[layer]
            sample_weight = self.sample_weight[layer]
            metrics_layer = dict(labels.apply(lambda x: self.eval_single_label(predictions[x],
                                                                               actual_sources[x],
                                                                               sample_weight)))
            metrics_layers.append(metrics_layer)

            sample_count_layer = actual_sources.sum()
            avg_labels = list(sample_count_layer[sample_count_layer > self.sample_count_threshold].index)
            avg_metrics = ['Acc', 'Sn', 'Sp', 'TPR', 'FPR', 'Rc', 'Pr', 'F1', 'F-max', 'ROC-AUC', 'PR-AUC']
            avg_metrics_layer = pd.DataFrame(np.concatenate( [np.expand_dims(metrics_layer[label][avg_metrics].to_numpy(), 2)
                                                 for label in avg_labels], axis=2).mean(axis=2), columns=avg_metrics) # here
            avg_metrics_layers.append(avg_metrics_layer)
        return metrics_layers, avg_metrics_layers

    def eval_single_label(self, predictions: pd.Series, actual_sources: pd.Series, sample_weight):
        label = actual_sources.name
        print('Evaluating biome source:', label)
        pred_source = (predictions.to_numpy().reshape(1, predictions.shape[0]) >= self.thresholds).astype(np.uint)
        pred_source = pd.DataFrame(pred_source, columns=predictions.index)
        metrics = pd.DataFrame()
        metrics['t'] = self.thresholds.flatten()
        actual_sources = actual_sources.astype(np.uint)
        conf_matrix = metrics['t'].apply(lambda T: confusion_matrix(actual_sources, pred_source.iloc[int(T * 200), :],
                                                                    sample_weight=sample_weight, labels=[0, 1]).ravel())
        conf_metrics = pd.DataFrame(conf_matrix.tolist(), columns=['TN', 'FP', 'FN', 'TP']).astype(np.int)
        metrics = pd.concat( (metrics, conf_metrics), axis=1).set_index('t')
        metrics['Acc'] = metrics[['TP', 'TN']].sum(axis=1) / metrics.sum(axis=1)
        metrics['Sn'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
        metrics['Sp'] = metrics['TN'] / (metrics['TN'] + metrics['FP'])
        metrics['TPR'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
        metrics['FPR'] = metrics['FP'] / (metrics['TN'] + metrics['FP'])
        metrics['Rc'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
        metrics['Pr'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
        metrics = metrics.fillna(1)
        metrics['F1'] = (2 * metrics['Pr'] * metrics['Rc'] / (metrics['Pr'] + metrics['Rc']))
        metrics['ROC-AUC'] = ((metrics['FPR'][:-1] + metrics['FPR'][1:]) * (metrics['TPR'][:-1] - metrics['TPR'][1:]) / 2).sum()
        metrics['PR-AUC'] = ((metrics['Pr'][:-1] + metrics['Pr'][1:]) * (metrics['Rc'][:-1] - metrics['Rc'][1:]) / 2).sum()
        metrics['F-max'] = metrics['F1'].max()
        print(metrics)
        return metrics