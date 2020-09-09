import requests
from pprint import pprint
from io import StringIO
import pandas as pd
from tqdm import trange


prefix = '''
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
PREFIX mesh2015: <http://id.nlm.nih.gov/mesh/2015/>
PREFIX mesh2016: <http://id.nlm.nih.gov/mesh/2016/>
PREFIX mesh2017: <http://id.nlm.nih.gov/mesh/2017/>
PREFIX mesh2018: <http://id.nlm.nih.gov/mesh/2018/>
PREFIX mesh2019: <http://id.nlm.nih.gov/mesh/2019/>
PREFIX mesh2020: <http://id.nlm.nih.gov/mesh/2020/>

SELECT ?treeNum ?ancestorTreeNum ?ancestor ?alabel
FROM <http://id.nlm.nih.gov/mesh>

WHERE {
'''

unit = '''
{{
   mesh:{0} meshv:treeNumber ?treeNum .
   ?treeNum meshv:parentTreeNumber+ ?ancestorTreeNum .
   ?ancestor meshv:treeNumber ?ancestorTreeNum .
   ?ancestor rdfs:label ?alabel .
}} 
UNION 
{{
   mesh:{0} meshv:treeNumber ?treeNum .
   mesh:{0} meshv:treeNumber+ ?ancestorTreeNum .
   ?ancestor meshv:treeNumber ?ancestorTreeNum .
   ?ancestor rdfs:label ?alabel .
}}
'''

ids = ['D003863', 'D012559', 'D001714', 'D043183', 'D003248', 'D013959', 'D003967', 
	   'D001327', 'D008881', 'D008171', 'D007410', 'D006262', 'D001289', 'D015212', 
	   'D002318', 'D003920', 'D003015', 'D000067877', 'D008107', 'D002446', 'D007674', 
	   'D004827', 'D000544', 'D010661']
units = [unit.format(id) for id in ids]

suffix = '''
FILTER (strStarts(str(?treeNum), str(?ancestorTreeNum)))
}
ORDER BY ?treeNum ?ancestorTreeNum
'''

url = 'https://id.nlm.nih.gov/mesh/sparql'
headers = {"Accept-Language": "en-US,en;q=0.5"}
df_tosave = pd.DataFrame(columns=['treeNum', 'ancestorTreeNum', 'ancestor', 'alabel'])

for i in trange(0, len(units), 10):
	query = prefix + 'UNION'.join(units[i:min(i+10, len(units))]) + suffix
	payload = {'query': query, 'format':'JSON', 'inference': 'false', 'offset': '0', 'limit': '1000'}
	get = requests.get(url, params=payload, headers=headers, timeout=10)
	df = pd.DataFrame(filter(lambda x: x['alabel']['xml:lang'] == 'en', get.json()['results']['bindings']))
	df = df.applymap(lambda x: x['value'].split('/')[-1] if x['value'].startswith('http') else x['value'])
	df_tosave = df_tosave.append(df.reset_index(drop=True), ignore_index=True)

print('common:', set(ids).intersection(set(df_tosave['ancestor'].tolist())))
print('difference:', set(ids) - set(df_tosave['ancestor'].tolist()))
df_tosave.to_csv('data_by_sparql.csv')

pprint(df_tosave)
