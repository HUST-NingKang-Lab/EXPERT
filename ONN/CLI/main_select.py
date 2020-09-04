from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import os


def select(args):
	matrix_genus = pd.read_hdf(args.i, key='genus')
	phylo = pd.read_csv(args.phylo, index_col=0)
	C = args.C
	print(matrix_genus)

	if args.filter_only:
		matrix_genus.loc[phylo['genus'], :].to_hdf(args.o, key='genus')
	elif args.use_rf:
		X = matrix_genus.T
		Y = pd.concat([pd.read_hdf(args.labels, key='l' + str(layer)) for layer in range(args.dmax)], axis=1)

		selector = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=args.p,
									  random_state=1, verbose=5)
		selector.fit(X, Y)
		print('Done')
		importance = selector.feature_importances_
		select_idx = importance > C * importance.mean()
		new_phylo = phylo.iloc[select_idx, :].reset_index(drop=True)
		print('New phylogeny:')
		print(new_phylo)
		matrix_genus.loc[new_phylo['genus'], :].to_hdf(args.o, key='genus')
		new_phylo.to_csv(os.path.join(args.tmp, 'phylogeny_selected_using_rf_importance_C{}.csv'.format(C)))
	else:
		X = matrix_genus.T
		variance = X.var(axis=0)
		selector = VarianceThreshold(threshold=C * variance.mean())
		selector.fit(X)
		select_idx = selector.get_support()
		new_phylo = phylo.iloc[select_idx, :].reset_index(drop=True)
		print('New phylogeny:')
		print(new_phylo)
		matrix_genus.loc[new_phylo['genus'], :].to_hdf(args.o, key='genus')
		new_phylo.to_csv(os.path.join(args.tmp, 'phylogeny_selected_using_varianceThreshold_C{}.csv'.format(C)))
