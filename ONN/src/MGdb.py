import os
import json
import urllib.request
import urllib.error
import requests


def set_proxy(proxy_dict):
	proxy = urllib.request.ProxyHandler(proxy_dict)
	opener = urllib.request.build_opener(proxy)
	urllib.request.install_opener(opener)


class MGnify(object):

	def __init__(self):
		self.url_prefix, status = self.json_url_as_dict('https://www.ebi.ac.uk/metagenomics/api/v1/')

	def get_study_obj(self, study):
		url = self.url_prefix['studies'] + '/' + study
		study_obj, status = self.json_url_as_dict(url)
		return study_obj

	def get_biome_obj(self, biome):
		url = self.url_prefix['biomes'] + '/' + biome
		biome_obj, status = self.json_url_as_dict(url)
		return biome_obj

	def get_analysis_obj(self, analysis_id):
		url = self.url_prefix['analyses'] + '/' + analysis_id
		analysis_obj, status = self.json_url_as_dict(url)
		return analysis_obj

	def get_sample_obj(self, sample_id):
		url = self.url_prefix['samples'] + '/' + sample_id
		sample_obj, status = self.json_url_as_dict(url)
		return sample_obj

	def taxassign_from_study(self, study_id):
		url = self.url_prefix['studies'] + '/{}/downloads'.format(study_id)
		downloads, status = self.json_url_as_dict(url)
		tax_assigns = {download['id']: download['links']['self'] for download in downloads if self.__is_taxassign_tsv(download)}
		return tax_assigns

	def sample_from_run(self, run_id):
		url = self.url_prefix['runs'] + '/{}'.format(run_id)
		run, status1 = self.json_url_as_dict(url)
		sample_link = run['relationships']['sample']['links']['related']
		#print(sample_link)
		sample, status2 = self.json_url_as_dict(sample_link)
		return sample, status1, status2

	def __is_taxassign_tsv(self, d):
		attr = d['attributes']
		if attr['group-type'] in {'Taxonomic analysis', 'Taxonomic analysis SSU rRNA'} and \
				attr['file-format']['extension'] == 'tsv' and \
				attr['description']['label'] in {'Taxonomic assignments',
												 'Taxonomic assignments SSU'}:
			return True
		else:
			return False

	def retrieve(self, link, filename):
		try:
			urllib.request.urlretrieve(link, filename)
		except urllib.error.ContentTooShortError:
			raise Warning('Object too short: {}'.format(link))
		except urllib.error.URLError:
			raise Warning('cannot open URL: {}'.format(link))

	def json_url_as_dict(self, url, use_request=False):
		if use_request:
			return requests.get(url).json()
		else:
			res = None
			while res == None:
				try:
					with urllib.request.urlopen(url) as url:
						res = json.loads(url.read().decode())['data']
				except(urllib.error.URLError, OSError):
					print('A problem has occured, still retrying....')
					pass
		return res, 'Pass'

	def _pthjoin(self, pth1, pth2):
		os.path.join(pth1, pth2)
