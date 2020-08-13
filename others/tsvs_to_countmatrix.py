from functools import reduce
import pandas as pd
import os


data_dir = 'data/'
biome_dirs = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
tsv_dirs = [os.path.join(biome_dir, tsv) for biome_dir in biome_dirs 
            for tsv in os.listdir(biome_dir)]

tsvs = map(lambda x: pd.read_csv(x, sep='\t', header=1), tsv_dirs)
'''for i in tsvs:
    print('taxonomy' in i.columns)'''
tidy_tsv = lambda x: x[x.columns[1:3]].groupby(by='taxonomy').sum()
tsvs = map(tidy_tsv, tsvs)
scale_to_1 = lambda x: x.apply(lambda x: (x / x.sum()) if x.name!='taxonomy' else x, axis=0)
tsvs = list(map(scale_to_1, tsvs))
print(tsvs[0])

cm = reduce(lambda x, y: pd.merge(left=x, right=y, 
            on=['taxonomy'], how='outer'), tsvs)
cm.fillna(0).T.to_csv('count_matrix.csv')