from expert.src.evaluator import Evaluator
import pandas as pd
import os
from expert.src.utils import get_dmax
from tqdm import trange
import numpy as np
from joblib import Parallel


def evaluate(cfg, args):
    layers = [os.path.join(args.i, i) for i in sorted(os.listdir(args.i), key=lambda x: int(x.split('.')[0].split('-')[1]))]
    np.random.seed(0)
    #idx = np.random.choice(np.arange(100000), 10000)
    predictions = [pd.read_csv(layer, index_col=0)#.iloc[idx, :]
                   for layer in layers]
    sources = [pd.read_hdf(args.labels, key='l'+str(layer))#.iloc[idx, :]
               for layer in range(get_dmax(args.labels))]
    if 'root' in sources[0].columns:
        sources = sources[1:]
    par = Parallel(n_jobs=args.p, prefer='threads')
    print('Running evaluation...')
    evaltr = Evaluator(predictions_multilayer=predictions, actual_sources_multilayer=sources,
                       num_thresholds=args.T, sample_count_threshold=100, par=par)
    metrics_layers, avg_metrics_layers = evaltr.eval()
    print('Saving evaluation results...')
    if not os.path.isdir(args.o):
        os.mkdir(args.o)
    for layer in trange(len(layers)):
        if not os.path.isdir(os.path.join(args.o, 'layer-' + str(layer))):
            os.mkdir(os.path.join(args.o, 'layer-' + str(layer)))
        metrics_layer = metrics_layers[layer]
        avg_metrics_layer = avg_metrics_layers[layer]
        avg_metrics_layer.to_csv(os.path.join(args.o, 'layer-' + str(layer) + '.csv' ))
        for label, metrics in metrics_layer.items():
            metrics.to_csv(os.path.join(args.o, 'layer-' + str(layer), label + '.csv'))

