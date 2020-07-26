from src.utils import *


# making biom, otus, tsvs into npz file. sparse matrix storage needed.

# checking data integrity

# keep all important information from raw data (paths, leaves of ontology and phylogeny tree).


class Transformer(object):

	def __init__(self, output_path, tmp_path, phylogeny=None, tsv_paths=None,
				 biom_path=None, cm_path=None, db_file=None):
		# provide tsv_paths as array or series, not list
		self.output_path = output_path
		self.tmp_path = tmp_path
		if tsv_paths:
			self.paths = pd.Series(tsv_paths)
			self.error_msg = self._chech_all_tsv()
			self.countmatrix = self.get_CM_from_tsvs()
		elif biom_path:
			self.biom_path = biom_path
			self.countmatrix = self.get_CM_from_biom(biom_path)
		elif cm_path:
			self.cm_path = cm_path
			self.countmatrix = self.load_countmatrix(cm_path)

		self.db_tool = NCBITaxa(db_file=db_file)
		self.phylogeny = phylogeny

	def _chech_all_tsv(self, sep='\t', header=1):
		paths = self.paths
		return pd.Series([self._check_tsv(path, sep=sep, header=header) for path in paths])

	def _check_tsv(self, path, sep='\t', header=1):
		"""
		Check for: 1. IOError, 2.  columns, 3. na values, 4. negative values
		:param path:
		:return:
		"""
		tsv = pd.read_csv(path, sep=sep, header=header)
		try:
			tsv = pd.read_csv(path, sep=sep, header=header)
		except IOError:
			print('Cannot read file properly.')
			return 'IOError (cannot read)'
		else:
			pass

		if tsv.shape[0] == 0:
			print('No value found in dataframe')
			return 'Empty table'
		elif tsv.isna().sum().sum() > 0:
			print('NA value(s) exist in dataframe')
			return 'NA value(s)'
		elif (tsv.iloc[:, 1] < 0).sum() > 0:
			print('Negative value(s) exist in dataframe')
			return '<0 value(s)'
		return ''

	def get_CM_from_tsvs(self, sep='\t', header=1):
		"""
		to count matrix
		:param kwgs:
		:return:
		"""
		raw_tsvs = map(lambda x: pd.read_csv(x, sep=sep, header=header),
				   self.paths[self.error_msg == ''])
		# summarise abundance for identical entries
		tsvs = map(lambda x: x[x.columns[1:3]].groupby(by=x.columns[2]).sum(), raw_tsvs)
		count_matrix = reduce(lambda x,y: pd.merge(left=x, right=y, on='taxonomy',
												   how='outer'), tsvs)
		count_matrix = count_matrix.fillna(0)
		return count_matrix

	def _extract_layers(self, count_matrix, included_ranks=None):
		"""
		Step 1: Dealing with entries not in db
		Step 2: Track lineage for each taxonomy and join with count matrix
		Step 3: Group count matrix by eack rank and count the abundance, then left join with phylogeny dataframe on rank
		Step 4: Fit in a Numpy 3-d array, then swap axes [rank, taxonomy, sampleid] into [sampleid, taxonomy, rank]
		:param tsvs:
		:return:
		"""
		if included_ranks == None:
			included_ranks = ['superkingdom', 'phylum', 'class', 'order',
							  'family', 'genus', 'species']

		# filter, taxonomies with sk in db are kept.
		taxas = pd.Series(count_matrix.index.to_list(), index=count_matrix.index)
		sks = taxas.apply(lambda x: x.split(';')[0].split('__')[1])
		#print(sks)
		sk_indb = self.db_tool.entries_in_db(sks)

		print('There will be {}/{} entries droped cause they are not '
			  'in NCBI taxanomy database'.format((~sk_indb).sum(), sk_indb.shape[0]))
		# Consider using more flexible filter to handle:
		# 	[sk__Mitochondria;k__;p__;c__;o__;f__;g__;s__metagenome]
		print(sks[~sk_indb])

		sk_indb.index = count_matrix.index
		cm_keep = count_matrix[sk_indb]

		'''cm_keep.index = count_matrix['taxonomy']
		cm_keep = cm_keep.drop(columns=['taxonomy'])'''
		sampleids = cm_keep.columns.tolist()
		# extract
		multi_entries = pd.Series(cm_keep.index.to_list(), index=cm_keep.index.to_list())
		lineages = self._track_lineages(multi_entries=multi_entries)

		# post process
		add_prefix = lambda x: ('sk' if x.name == 'superkingdom'
								else x.name[0:1]) + '__' + x
		str_cumsum = lambda x: pd.Series([';'.join(x[0:i])
										  for i in range(1, x.shape[0] + 1)], index=x.index)
		lineages = lineages.apply(add_prefix, axis=0).apply(str_cumsum, axis=1)

		if self.phylogeny == None:
			# what about feature selection????????????????
			print('No default phylogeny tree provided, '
				  'will use all lineages data involved automatically.')
			self._updata_phylo(lineages)
			self.phylogeny.to_csv(self.get_savepath('phyogeny_tree.csv', type='out'))

		# join by index
		cm_with_lngs = cm_keep.join(lineages)
		cm_with_lngs.to_csv(self.get_savepath('countmatrix_with_lineages.csv', type='tmp'))

		# fit in phylogeny dataframe !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		fill_in_phylogeny = lambda x: pd.merge(left=self.phylogeny[[x]].copy(),
			right=cm_with_lngs.groupby(by=x, as_index=False).sum(), on=[x], how='left',
			suffixes=('_x','_y')).set_index(self.phylogeny[x])[sampleids]
		cm_ranks = OrderedDict( zip(included_ranks, map(fill_in_phylogeny, included_ranks) ))
		'''for key, df in cm_ranks.items(): df.to_csv(key+'.csv') # to_hdf5

		# key -> ranks, index -> taxonomies, column name -> sample ids
		the_Matrix = np.array(list(cm_ranks.values()))
		print(the_Matrix.shape)
		the_Matrix = the_Matrix.swapaxes(0, 2)  # transpose 3-d matrix'''
		return cm_ranks

	def _track_lineages(self, multi_entries):
		"""
		Already filtered, at least one entry of each taxonomy is (in entries) in db.
		Overview: entries name -> entries id -> entries lineage ids -> entries lineage names
		step 1: get_ids_from_names
		step 2: get_lineages_ids (get_ranks)
		step 3: get_names_from_ids
		:param entries:
		:return:
		"""
		entries = self._fathest_entry_in_db(multi_entries)                                 # series
		#print(entries[21])
		taxids = self.db_tool.get_ids_from_names(entries.tolist())
		lineages_ids = self.db_tool.get_lineage(taxids)                                    # dataframe, fillna?
		lineages_ids.index = entries.index
		#names = lineages_ids.apply(self.db_tool.get_names_from_ids, axis=1)                # prefix ??????????????
		id2name = lambda id: self.db_tool.get_names_from_ids([id])[0] if id != 0 else ''
		names = lineages_ids.fillna(0).applymap(int).applymap(id2name)
		names.to_csv(self.get_savepath('lineage_names.csv', type='tmp'))
		# lineages_ids has many many nan values, this need to be fixed.
		# considering using element-wise applymap
		return names

	def _fathest_entry_in_db(self, multi_entries):
		"""
		already filtered, all entries are contained in db.
		:param multi_entries:
		:return:
		"""
		entries_se = multi_entries.str.split(';').apply(lambda x: {i.split('__')[0]:
																	   i.split('__')[1] for i in x})
		# get tidy data
		entries_df = pd.DataFrame(entries_se.tolist(), index=entries_se.index).fillna('')
		entries_df = entries_df.applymap(lambda x: x.replace('_', ' '))

		isin_db = entries_df.apply(self.db_tool.entries_in_db, axis=1)
		isfarthest_indb = pd.DataFrame(isin_db.apply(lambda x: x.index == x[x].index[-1],
													 axis=1).values.tolist(),
									   index=entries_df.index,
									   columns=entries_df.columns)

		farthest_entries = pd.Series(entries_df[isfarthest_indb].fillna('').\
			apply(str_sum, axis=1).values.tolist(), index=entries_df.index)

		return farthest_entries

	def get_label(self, ids_by_layer):
		env_sample = self.paths[self.error_msg == ''].str.split('/').apply(lambda x: x[-2:])
		labels = pd.DataFrame(env_sample.tolist(), columns=['Envs', 'SampleID'],
							  index=env_sample.index)
		idx = labels['SampleID'].str.extract(pat=r'([A-Z]RR.[0-9]{1,})', expand=False)
		labels = labels.set_index(idx).drop(columns=['SampleID'])

		nlayers = len(ids_by_layer)
		layers = 'layer_' + pd.Series(map(str, range(1, nlayers + 1)))

		str_cumsum = lambda x: [':'.join(x[0:i]) for i in range(1, len(x) + 1)]
		extract_layers = lambda x: str_cumsum(x.split(':') + ['Unknown'] *
									(nlayers - 1 - x.count(':')))
		#print(labels['Envs'].apply(extract_layers).tolist())
		extd_layers = pd.DataFrame(labels['Envs'].apply(extract_layers).tolist(),
								   columns=layers, index=labels.index)
		labels = labels.drop(columns=['Envs']).join(extd_layers)
		print(labels)
		ids_by_layer = {layer: pd.Series(ids) for layer, ids in ids_by_layer.items()}
		for layer in ids_by_layer.keys():
			col = 'layer_'+str(layer)
			labels[col] = labels[col].apply(lambda x: (x == ids_by_layer[layer]).
											astype(int).tolist())
		return labels

	def get_savepath(self, file_name, type='tmp'):
		if type == 'tmp':
			if not os.path.isdir(self.tmp_path):
				os.mkdir(self.tmp_path)
			os.path.join(self.tmp_path, file_name)
		elif type == 'out':
			if not os.path.isdir(self.output_path):
				os.mkdir(self.output_path)
			os.path.join(self.output_path, file_name)
		else:
			print("Please give the correct `type`: {'tmp','out'}")

	def _updata_phylo(self, lineage_names):
		self.phylogeny = lineage_names

	def get_CM_from_biom(self, biom_path):
		pass

	def load_countmatrix(self, cm_path):
		pass


class NCBITaxa(object):

	def __init__(self, db_file=None, in_memory=True):
		if db_file == None:
			self.db_file = '/root/.etetoolkit/taxa.sqlite'
		else:
			self.db_file = db_file
		self.db = self._get_db(in_memory=in_memory)

	def _get_db(self, in_memory=True):
		source = sqlite3.connect(self.db_file)
		if in_memory:
			dest = sqlite3.connect(':memory:')
			source.backup(dest)
			return dest
		else:
			return source

	def entries_in_db(self, entries: pd.Series):
		joined_names = ','.join(entries.apply(lambda x: '"'+x+'"'))
		command = 'select spname, taxid FROM species WHERE spname IN ({})'.format(joined_names)
		name = {name for name, _ in self.db.execute(command).fetchall()}
		if name == set([]):
			in_db = pd.Series(False, index=entries.index)
			return in_db
		ret = entries.isin(name)
		return ret

	def get_lineage(self, ids, include_ranks=None):
		if include_ranks == None:
			include_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family',
							 'genus', 'species']
		str_ids = map(str, ids)
		command = 'select taxid, track FROM species WHERE taxid IN ({})'.format(','.join(str_ids))
		id2lineage = {id: lineage for id, lineage in self.db.execute(command).fetchall()}
		lineages = pd.Series([id2lineage[id].split(',') for id in ids])
		ranks2lineages = lineages.apply(lambda x: dict(zip(self.get_ranks_from_ids(x), x) ) )
		# simpler below
		#print(ranks2lineages)
		table = pd.DataFrame(ranks2lineages.tolist(),
							 index=np.arange(ranks2lineages.shape[0]),
							 columns=include_ranks)
		#print(table)
		table.to_csv('lineages.csv')
		''':
			table = table.append(pd.DataFrame(track, index=[0]))'''
		table.fillna('__')
		#print(table.columns)
		#print(include_ranks)
		# return dataframe here !!!!!
		return table

	def get_ranks_from_ids(self, ids):
		str_ids = map(str, ids)
		command = 'select taxid, rank FROM species WHERE taxid IN ({})'.format(','.join(str_ids))
		id2rank = {id: rank for id, rank in self.db.execute(command).fetchall()}
		ranks = [id2rank[int(id)] for id in ids]
		return ranks

	def get_ids_from_names(self, names):
		"""
		Problem to be solved: Non-unique ids for each name.
		:param names:
		:return:
		"""
		joined_names = ','.join( map( lambda x: '"'+x+'"', names) )
		command = 'select spname, taxid FROM species WHERE spname IN ({})'.format(joined_names)
		name2id = {name: id for name, id in self.db.execute(command).fetchall()}
		names = [name2id[name] for name in names]
		return names

	def get_names_from_ids(self, ids):
		str_ids = map(str, ids)
		command = 'select taxid, spname FROM species WHERE taxid IN ({})'.format(','.join(str_ids))
		id2name = {id: name for id, name in self.db.execute(command).fetchall()}
		names = [id2name[id] for id in ids]
		return names


'''names = ['bacteria']
command = 'select spname, taxid FROM species WHERE spname IN ({})'.format(','.join(map( lambda x:'\"'+x+'\"', names)))
name2id = {name: id for name, id in db.execute(command).fetchall()}
db.execute(command).fetchall()'''


pths = [os.path.join('../../data/root:Mixed', i)
		for i in os.listdir('../../data/root:Mixed')]
t = Transformer(tsv_paths=pths, output_path='out', tmp_path='tmp')
#cm = t.get_CM_from_tsvs()
ids_by_layer = {1: ['root'], 2:['root:Mixed', 'root:Host-associated']}


class Selector(object):

	def __init__(self, CountMatrix, n_jobs=1, max_depth=10, verbose=5):
		"""
		:param CountMatrix: already cleaned count matrix, all entries are contained in db,
			using consensus lineage as index
		:param n_jobs:
		:param max_depth:
		:param verbose:
		"""
		self.countmatrix = CountMatrix
		self.rf = RandomForestRegressor(random_state=1, max_depth=max_depth,
										n_jobs=n_jobs, verbose=verbose)
		self.importance = np.zeros((CountMatrix.shape[0], ))
		self.phylogeny = pd.Series(CountMatrix.index.to_list())

	def cal_importance(self, X, Y):
		self.rf.fit(X, Y)
		self.importance = self.rf.feature_importances_

	def select_index(self, coefficient=1e-3):
		keep_idx = self.importance >= (self.importance.mean() * coefficient)
		return keep_idx

	def save_index(self, index, path):
		np.save(path, index)









