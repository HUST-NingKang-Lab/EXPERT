from .utils import *

# making biom, otus, tsvs into npz file. sparse matrix storage needed.

# checking data integrity

# keep all important information from raw data (paths, leaves of ontology and phylogeny tree).


# core manipulating process
feature_df = pd.DataFrame() # feature_dataframe: 8 * ..feature number
ranks = ['sk', 'k', 'p', 'c', 'o', 'f', 'g', 's']
samples = [pd.DataFrame()]
extract_lineages = lambda x: x
for sample in samples:  # can be parallel by threading, because groupby and disk IO releases the GIL.
	df = extract_lineages(sample)  # 9 columns
	dfs = [df] + [df.groupby(rank).sum()[rank, 'abu'] for rank in ranks] # [rank, abu]
	abus = reduce(lambda x, y: pd.merge(left=x, right=y, on=y.columns[0], suffixes=('', y.columns[0])), dfs)
	sample_features = pd.merge(left=feature_df, right=abus, on=ranks, suffixes=('', ''))
	# save_to_file


class Transformer(object):

	def __init__(self, phylogeny=None, tsv_paths=None, biom_path=None, otus_path=None, **kwgs):
		# provide tsv_paths as array or series, not list
		if tsv_paths:
			self.paths = tsv_paths
			self.error_msg = self._chech_all_tsv(**kwgs)
		self.db_tool = NCBITaxa(db_file=kwgs['db_file'])
		self.phylogeny = phylogeny

	def _chech_all_tsv(self, **kwgs):
		paths = self.paths
		return pd.Series([self._check_tsv(path, **kwgs) for path in paths])

	def _check_tsv(self, path, **kwgs):
		"""
		Check for: 1. IOError, 2.  columns, 3. na values, 4. negative values
		:param path:
		:return:
		"""
		try:
			tsv = pd.read_csv(path, sep=kwgs['sep'], header=kwgs['header'])
		except IOError:
			print('Cannot read file properly.')
			return 'IOError (cannot read)'

		if tsv.shape[0] == 0:
			print('No value found in dataframe')
			return 'Empty table'
		elif tsv.isna().sum() > 0:
			print('NA value(s) exist in dataframe')
			return 'NA value(s)'
		elif (tsv.iloc[:, 1] < 0).sum() > 0:
			print('Negative value(s) exist in dataframe')
			return '<0 value(s)'
		return ''

	def get_CM_from_tsvs(self, **kwgs):
		"""
		to count matrix
		:param kwgs:
		:return:
		"""
		raw_tsvs = map(lambda x: pd.read_csv(x, sep=kwgs['sep'], header=kwgs['header']),
				   self.paths[self.error_msg == ''])
		# summarise abundance for identical entries
		tsvs = map(lambda x: x[x.columns[1:2]].groupby(by=x.columns[2]).sum(), raw_tsvs)
		count_matrix = reduce(lambda x,y: pd.merge(left=x, right=y, on='taxonomy'), tsvs)
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
			included_ranks = include_ranks = ['sk', 'k', 'p', 'c', 'o', 'f', 'g']

		# filter, taxonomies with sk in db are kept.
		sks = count_matrix['taxonomy'].str.lower().str.extract(pat=r'((?<=sk__)\w{1,}(?=\b|;k))', expand=False)
		cm_keep = count_matrix[self.db_tool.entries_in_db(sks)]
		cm_keep.index = count_matrix['taxonomy']
		cm_keep = cm_keep.drop(columns=['taxonomy'])

		# extract
		lineages = self._track_lineages(multi_entries=cm_keep.index.to_series())

		# join by index
		cm_with_lngs = cm_keep.join(lineages)

		# fit in phylogeny dataframe
		fill_in_phylogeny = lambda x: self.phylogeny[x].copy().join(cm_with_lngs.groupby(by=x, as_index=False).sum(),
																	on=x, how='left')
		cm_ranks = OrderedDict( zip(included_ranks, map(fill_in_phylogeny, included_ranks) ))
		# key -> ranks, index -> taxonomies, column name -> sample ids
		RSRADM = np.array(cm_ranks.values()).swapaxes(axis1=0, axis2=2)                    # transpose 3-d matrix
		return RSRADM

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
		taxids = self.db_tool.get_ids_from_names(entries.tolist())
		lineages_ids = self.db_tool.get_lineage(taxids)                                    # dataframe, fillna?
		names = lineages_ids.apply(self.db_tool.get_names_from_ids, axis=1)                # prefix ??????????????
		# dataframe ?
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
		entries_df = pd.DataFrame(entries_se.tolist()).fillna('')
		entries_df.index = entries_se.index                            # changed here
		entries_df = entries_df.applymap(lambda x: x.replace('_', ' '))
		# disgards entries from Chloroplast and Mitochondria
		'''disgards = ['Chloroplast', 'chloroplast', 'Mitochondria', 'mitochondria']
		entries_df = entries_df[~entries_df['sk'].str.contains('|'.join(disgards))].reset_index(drop=True)'''
		isin_db = entries_df.apply(self.db_tool.entries_in_db, axis=1)
		isfarthest_indb = pd.DataFrame(isin_db.apply(lambda x: x.index == x[x].index[-1], axis=1).tolist(),
									   columns=entries_df.columns)
		isfarthest_indb.index = entries_se.index
		farthest_entries = entries_df[isfarthest_indb].fillna('').apply(str_sum, axis=1)
		return farthest_entries


class NCBITaxa(object):

	def __init__(self, db_file=None, in_memory=True):
		if db_file == None:
			self.db_file = '/root/.etetoolkit/taxa.sqlite'
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
		ret = entries.str.contains('|'.join(name))
		return ret

	def get_lineage(self, ids, include_ranks=None):
		if include_ranks == None:
			include_ranks = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family',
							 'genus', 'species']

		command = 'select track FROM species WHERE taxid IN ({})'.format(','.join(ids))
		id2lineage = {id: lineage for id, lineage in self.db.execute(command).fetchall()}
		lineages = pd.Series([id2lineage[id] for id in ids])
		ranks2lineages = lineages.apply(lambda x: dict(zip(self.get_ranks_from_ids(x), x) ) )
		# simpler below
		table = pd.DataFrame()
		for track in ranks2lineages:
			table = table.append(pd.DataFrame(track, index=[0]))
		table.fillna('__')

		# return dataframe here !!!!!
		return table[include_ranks].values.tolist()

	def get_ranks_from_ids(self, ids):
		command = 'select taxid, rank FROM species WHERE taxid IN ({})'.format(','.join(ids))
		id2rank = {id: rank for id, rank in self.db.execute(command).fetchall()}
		ranks = [id2rank[id] for id in ids]
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
		command = 'select taxid, spname FROM species WHERE taxid IN ({})'.format(','.join(ids))
		id2name = {id: name for id, name in self.db.execute(command).fetchall()}
		names = [id2name[id] for id in ids]
		return ids


'''names = ['bacteria']
command = 'select spname, taxid FROM species WHERE spname IN ({})'.format(','.join(map( lambda x:'\"'+x+'\"', names)))
name2id = {name: id for name, id in db.execute(command).fetchall()}
db.execute(command).fetchall()'''