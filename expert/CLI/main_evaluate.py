from expert.src.evaluator import Evaluator
import pandas as pd
import os
from expert.src.utils import get_dmax
from tqdm import trange


def evaluate(cfg, args):
    layers = [os.path.join(args.i, i) for i in sorted(os.listdir(args.i), key=lambda x: int(x.split('.')[0].split('-')[1]))]
    predictions = [pd.read_csv(layer, index_col=0) for layer in layers]
    sources = [pd.read_hdf(args.labels, key='l'+str(layer)) for layer in range(get_dmax(args.labels))]
    print('Running evaluation...')
    evaltr = Evaluator(predictions_multilayer=predictions, actual_sources_multilayer=sources)
    metrics_layers, avg_metrics_layers = evaltr.eval()
    print('Saving evaluation results...')
    for layer in trange(len(layers)):
        metrics_layer = metrics_layers[layer]
        avg_metrics_layer = avg_metrics_layers[layer]
        avg_metrics_layer.to_csv(os.path.join(args.o, 'layer-' + str(layer) + '.csv' ))
        for label, metrics in metrics_layer.items():
            metrics.to_csv(os.path.join(args.o, 'layer-' + str(layer), label + '.csv'))

