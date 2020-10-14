from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import os


def select(cfg, args):
	matrix_genus = pd.read_hdf(args.i, key='genus')
	phylo = pd.read_csv(args.phylo, index_col=0)
	C = args.C
	print(matrix_genus)

	if not args.use_var and not args.use_rf:
		cols = set(phylo.columns.tolist())
		mat = phylo.set_index('genus').join(matrix_genus).fillna(0)
		mat = mat.drop(columns= (cols - {'genus'}) )
		print(mat.head())
		mat.to_hdf(args.o, key='genus')
	elif args.use_rf:
		X = (matrix_genus / matrix_genus.sum()).T  # abundance
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
	elif args.use_var:
		X = (matrix_genus / matrix_genus.sum()).T
		variance = X.var(axis=0)
		selector = VarianceThreshold(threshold=C * variance.mean())
		selector.fit(X)
		select_idx = selector.get_support()
		new_phylo = phylo.iloc[select_idx, :].reset_index(drop=True)
		print('New phylogeny:')
		print(new_phylo)
		matrix_genus.loc[new_phylo['genus'], :].to_hdf(args.o, key='genus')
		new_phylo.to_csv(os.path.join(args.tmp, 'phylogeny_selected_using_varianceThreshold_C{}.csv'.format(C)))
	else:
		raise InterruptedError('Please specify `-use-var` or `-use-rf` or none of them. See GitHub for details.')