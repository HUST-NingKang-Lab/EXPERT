from .utils import str_sum
import sqlite3
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import os
import gc
import numpy as np


# making biom, otus, tsvs into npz file. sparse matrix storage needed.
# keep all important information from raw data (paths, leaves of ontology and phylogeny tree).


class Transformer(object):

	def __init__(self, conf_path, phylogeny, db_file):
		self.conf_path = conf_path
		self.db_tool = NCBITaxa(db_file=db_file)
		self.phylogeny = phylogeny

	def _extract_layers(self, count_matrix, included_ranks=None, verbose=10):
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
							  'family', 'genus']

		# Filter, taxonomies with sk in db are kept.
		taxas = pd.Series(count_matrix.index.to_list(), index=count_matrix.index)
		#with open(self.get_conf_savepath('taxas.txt'), 'w') as f:
			#f.write('\n'.join(taxas))
		sks = taxas.apply(lambda x: x.split(';')[0].split('__')[1])
		sk_indb = self.db_tool.entries_in_db(sks)
		if verbose > 0:
			print('There will be {}/{} entries droped cause they are not '
				  'in NCBI taxanomy database'.format((~sk_indb).sum(), sk_indb.shape[0]))
			print(sks[~sk_indb])

		# Change index for boolean indeces
		sk_indb.index = count_matrix.index
		cm_keep = count_matrix[sk_indb]
		cm_keep = (cm_keep / cm_keep.sum()).astype(np.float32)
		
		del count_matrix, taxas, sks, sk_indb
		gc.collect()

		# Extract layers for entries data
		if verbose > 0:
			print('Extracting lineages for taxonomic entries, this may take a few minutes')
		multi_entries = pd.Series(cm_keep.index.to_list(), index=cm_keep.index.to_list())
		lineages = self._track_lineages(multi_entries=multi_entries)

		# Post-process lineages
		add_prefix = lambda x: ('sk' if x.name == 'superkingdom'
								else x.name[0:1]) + '__' + x
		str_cumsum = lambda x: pd.Series([';'.join(x[0:i])
										  for i in range(1, x.shape[0] + 1)], index=x.index)
		lineages = lineages.apply(add_prefix, axis=0).apply(str_cumsum, axis=1)

		del multi_entries
		gc.collect()

		# Fill samples in phylogeny dataframe
		if verbose > 0:
			print('Filling samples in phylogeny matrix')
		sampleids = cm_keep.columns.tolist()
		fill_in_phylogeny = lambda x: pd.merge(left=self.phylogeny[[x]].copy(),
			right=cm_with_lngs.groupby(by=x, as_index=False).sum(), on=[x], how='left',
			suffixes=('_x','_y')).set_index(x)[sampleids].fillna(0)
		
		# Setting genus as index
		cm_with_lngs = cm_keep.join(lineages)
		cm_with_lngs = cm_with_lngs.groupby(by=included_ranks, sort=False, as_index=False).sum()
		
		del cm_keep
		gc.collect()
		
		if self.phylogeny is not None:
			if verbose > 0:
				print('Generating matrix for each rank')
			# join by index
			matrix_by_rank = OrderedDict( zip(included_ranks, map(fill_in_phylogeny, tqdm(included_ranks)) ))
			# key -> ranks, index -> taxonomies, column name -> sample ids
			print(matrix_by_rank['genus'].describe(percentiles=[]))
			return matrix_by_rank
		else:
			if verbose > 0:
				print('No default phylogeny tree provided, '
					  'use all lineages data involved automatically.')
			self._updata_phylo(lineages)
			matrix_genus = fill_in_phylogeny('genus')
			print(matrix_genus.describe(percentiles=[]))
			#print(matrix_genus.sum())
			return matrix_genus

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
		#names.to_csv(self.get_savepath('lineage_names.tsv', type='tmp'), sep='\t')
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

	def get_conf_savepath(self, file_name):
		return os.path.join(self.conf_path, file_name)

	def _updata_phylo(self, lineage_names):
		lineage_names = lineage_names[['superkingdom','phylum','class','order','family','genus']]
		print('Updating phylo: Just keeping Superkingdom to Genus for phylogeny: {}.'.format(lineage_names.shape))
		lineage_names = lineage_names.drop_duplicates(subset=['genus'], ignore_index=True)
		print('Updating phylo: After droping duplicates: {}.'.format(lineage_names.shape))
		self.phylogeny = lineage_names


class NCBITaxa(object):

	def __init__(self, db_file=None, in_memory=True):
		if in_memory:
			print('Initializing in-memory taxonomy database for ultra-fast querying.')
		else:
			print('Initializing on-disk taxonomy database, consider using in-memory mode to speed up.')
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
		command1 = 'select spname, taxid FROM species WHERE spname IN ({})'.format(joined_names)
		name1 = {name for name, _ in self.db.execute(command1).fetchall()}
		missing = set(entries.tolist()) - name1
		if missing:
			joined_missing = ','.join(pd.Series(missing).apply(lambda x: '"' + x + '"'))
			command2 = 'select spname, taxid from synonym where spname IN ({})'.format(joined_missing)
			name2 = {name for name, _ in self.db.execute(command2).fetchall()}
		else:
			name2 = {}
		name = name1.union(name2)
		if name == set([]):
			in_db = pd.Series(False, index=entries.index)
			return in_db
		ret = entries.isin(name)
		return ret

	def get_lineage(self, ids, include_ranks=None):
		if include_ranks == None:
			include_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family',
							 'genus']
		str_ids = map(str, ids)
		command = 'select taxid, track FROM species WHERE taxid IN ({})'.format(','.join(str_ids))
		id2lineage = {id: lineage for id, lineage in self.db.execute(command).fetchall()}
		lineages = pd.Series([id2lineage[id].split(',') for id in ids])
		ranks2lineages = lineages.apply(lambda x: dict(zip(self.get_ranks_from_ids(x), x) ) )
		# simpler below
		# print(ranks2lineages)
		table = pd.DataFrame(ranks2lineages.tolist(),
							 index=np.arange(ranks2lineages.shape[0]),
							 columns=include_ranks)
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
		missing = set(names.tolist()) - set(name2id.keys())
		if missing:
			joined_missing = ','.join(map(lambda x: '"{}"'.format(x), missing))
			command2 = 'select spname, taxid from synonym where spname IN ({})'.format(joined_missing)
			name2id_missing = {name: id for name, id in self.db.execute(command2).fetchall()}
		else:
			name2id_missing = {}
		name2id = {**name2id, **name2id_missing}
		names = [name2id[name] for name in names]
		return names

	def get_names_from_ids(self, ids):
		str_ids = map(str, ids)
		command = 'select taxid, spname FROM species WHERE taxid IN ({})'.format(','.join(str_ids))
		id2name = {id: name for id, name in self.db.execute(command).fetchall()}
		names = [id2name[id] for id in ids]
		return names


"""
if missing:
	query = ','.join(['"%s"' %n for n in missing])
	result = self.db.execute('select spname, taxid from synonym where spname IN (%s)' %query)
	for sp, taxid in result.fetchall():
		oname = name2origname[sp.lower()]
		name2id.setdefault(oname, []).append(taxid)
		#name2realname[oname] = sp
"""


'''names = ['bacteria']
command = 'select spname, taxid FROM species WHERE spname IN ({})'.format(','.join(map( lambda x:'\"'+x+'\"', names)))
name2id = {name: id for name, id in db.execute(command).fetchall()}
db.execute(command).fetchall()'''

'''
pths = [os.path.join('../../data/root:Mixed', i)
		for i in os.listdir('../../data/root:Mixed')]
t = Transformer(tsv_paths=pths, output_path='out', tmp_path='tmp')
#cm = t.get_CM_from_tsvs()
ids_by_layer = {1: ['root'], 2:['root:Mixed', 'root:Host-associated']}'''









