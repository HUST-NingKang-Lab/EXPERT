from joblib import Parallel, delayed
import pandas as pd
import os
from ONN.src.utils import read_input_list, runid_from_taxassign, format_sample_info # biome, meta, desc
from ONN.src.MGdb import MGnify
from tqdm import tqdm, trange
from functools import reduce
import urllib.request


def download(args):
	print('Setting proxies...')
	#proxy_dict = {'http': 'http://127.0.0.1:4780', 'https': 'https://127.0.0.1:4780'} # use customizable options
	#proxy_dict = {}
	#proxy = urllib.request.ProxyHandler(proxy_dict)
	#opener = urllib.request.build_opener(proxy)
	#opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, '
	#					  'like Gecko) Chrome/35.0.1916.153 Safari/537.36 SE 2.X MetaSr 1.0')]
	#urllib.request.install_opener(opener)
	out_dir = args.o
	mg = MGnify()
	par = Parallel(n_jobs=args.p, prefer='threads')
	find_and_format_sample = lambda x: format_sample_info(mg.sample_from_run(x))
	studies = read_input_list(args.i)
	print(studies)
	print('Retrieving studies...')
	tax_assigns = par(delayed(mg.taxassign_from_study)(study) for study in studies)
	#print(tax_assigns)
	tax_assigns = reduce(lambda x, y: {**x, **y}, tax_assigns).items()
	#print(tax_assigns)
	tax_assigns = pd.DataFrame(tax_assigns, columns=['Name', 'Url'])
	#print(tax_assigns)
	tax_assigns['Name'] = tax_assigns['Name'].apply(lambda x: os.path.join(out_dir, x))
	print('Retrieving taxonomy assignments analysis results...')
	par(delayed(mg.retrieve)(tax_assigns.loc[i, 'Url'], tax_assigns.loc[i, 'Name'])
		for i in trange(tax_assigns.shape[0]))
	runids_2d = map(runid_from_taxassign, tqdm(tax_assigns['Name']))
	filenames = tax_assigns['Name']
	print('Retrieving metadata for runs...')
	for index, runids_1d in enumerate(runids_2d):
		samples = pd.DataFrame(par(delayed(find_and_format_sample)(runid) for runid in tqdm(runids_1d)))
		samples['RunID'] = runids_1d
		samples.to_csv(filenames[index].split('_')[0]+'_sample_information.tsv', sep='\t')


