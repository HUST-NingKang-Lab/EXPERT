from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np


def select(args):
	matrix_genus = pd.read_hdf(args.cm, key='genus')
	phylo = pd.read_csv(args.i, index_col=0)
	#print(phylo)
	labels = [pd.read_hdf(args.labels, key='l'+str(layer)) for layer in range(args.dmax+1)]

	Y = pd.concat(labels, axis=1)
	X = matrix_genus.T
	print(Y)
	print(X)
	print('Fitting data in RandomForestRegressor...')
	model = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=args.p,
								  random_state=1, verbose=5)
	model.fit(X, Y)
	print('Done')
	importance = model.feature_importances_
	n = args.top
	select_idx = np.argpartition(importance, -n)[-n:]
	new_phylo = phylo.iloc[select_idx, :].reset_index(drop=True)
	print('New phylogeny:')
	print(new_phylo)
	new_phylo.to_csv(args.o)
