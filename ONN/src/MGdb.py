import os
import json
import urllib.request
from collections import namedtuple


def set_proxy(proxy_dict):
	proxy = urllib.request.ProxyHandler(proxy_dict)
	opener = urllib.request.build_opener(proxy)
	urllib.request.install_opener(opener)

class MGnify(object):

	def __init__(self, proxy_dict=None):
		self.url_prefix = self.json_as_obj(self.json_str_from_url('https://www.ebi.ac.uk/metagenomics/api/v1/'))


	def get_study_obj(self, study):
		pass

	def get_biome_obj(self, biome):
		pass

	def get_analysis_obj(self, ):
		pass

	def json_as_obj(self, json_str):
		json_obj_hook = lambda d: namedtuple('X', d.keys())(*d.values())
		return json.loads(json_str, object_hook=json_obj_hook)

	def json_str_from_url(self, url):
		with urllib.request.urlopen(url) as url:
			return url.read().decode()

	def _pthjoin(self, pth1, pth2):
		os.path.join(pth1, pth2)
