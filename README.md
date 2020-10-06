# EXPERT

**EXPERT**: an exact and pervasive expert model for source tracking based on transfer learning. Based on Ontology-aware Neural Network architecture, EXPERT was trained using information-weighted loss to selectively learn from the training data, thus bypassing the negative impact of incomplete biome source information of data. Transfer learning technique was used to better adapt to novel biome ontology in a timely manner. 

The program is designed to help you to transfer Ontology-aware Neural Network model to other source tracking tasks. 

## Support 

Feel free to contact us if you have any question. Thank you for using Ontology-aware neural network.

## Deployment

```shell script
python setup.py install
```

## Input formats

Generally, EXPERT takes two kinds of `abundance data` and a `mapper file` (when transferring to other tasks) as inputs.

### Taxonomic assignments result for a single sample (OTU table)

Notice that here is a header "# Constructed from biom file" in the first line.

<table><thead><tr><th colspan="3"># Constructed from biom file</th></tr></thead><tbody><tr><td># OTU ID</td><td>ERR1754760</td><td>taxonomy</td></tr><tr><td>207119</td><td>19.0</td><td>sk__Archaea</td></tr><tr><td>118090</td><td>45.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__;o__Nitrosopumilales;f__Nitro...</td></tr><tr><td>153156</td><td>38.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__;o__Nitrosopumilales;f__Nitro...</td></tr><tr><td>131704</td><td>1.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__Nitrososphaeria;o__Nitrososp...</td></tr><tr><td>103181</td><td>5174.0</td><td>sk__Bacteria</td></tr><tr><td>157361</td><td>9.0</td><td>sk__Bacteria;k__;p__;c__;o__;f__;g__;s__agricultural_soil_bacterium_SC-I-11</td></tr></tbody></table>

### Taxonomic assignments result for multiple samples (count matrix)

<table><thead><tr><th>#SampleID</th><th>ERR1844510</th><th>ERR1844449</th><th>ERR1844450</th><th>ERR1844451</th></tr></thead><tbody><tr><td>sk__Archaea</td><td>1.0</td><td>17.0</td><td>8.0</td><td>16.0</td></tr><tr><td>sk__Archaea;k__;p__Crenarchaeota</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>sk__Archaea;k__;p__Euryarchaeota</td><td>8.0</td><td>2.0</td><td>3.0</td><td>1.0</td></tr><tr><td>sk__Archaea;k__;p__Eury...;c__...</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>sk__Archaea;k__;p__Eury...;c__...;o__...</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></tbody></table>

### Mapper file

<table><thead><tr><th></th><th>Env</th><th>SampleID</th></tr></thead><tbody><tr><td>0</td><td>root:Engineered:Wastewater</td><td>ERR2260442</td></tr><tr><td>1</td><td>root:Engineered:Wastewater</td><td>SRR980322</td></tr><tr><td>2</td><td>root:Engineered:Wastewater</td><td>ERR2985272</td></tr><tr><td>3</td><td>root:Engineered:Wastewater</td><td>ERR2814648</td></tr><tr><td>4</td><td>root:Engineered:Wastewater</td><td>ERR2985275</td></tr></tbody></table>

## Usage



## Demo

Getting a list of input files.



Constructing microbiome ontology using microbiomes.txt.

```shell script
ONN construct -i tmp/microbiomes.txt -o tmp/ontology.pkl
```

Getting mapper file from top level directory of well-organized data.

```shell script
ONN map -from-dir -i ../data/ -o tmp/mapper.csv
```

Mapping their source environments to microbiome ontology. Getting Hierarchical labels of all samples. 

```shell script
ONN map -to-otlg -otlg tmp/ontology.pkl -i tmp/mapper.csv -o tmp/out/labels.h5 -unk
```

Converting input data to a count matrix at genus level and generating phylogeny using the taxonomic entries data, Prepare for selecting top n important entries.

```shell script
ONN convert -i tmp/sample-list.txt -tmp tmp -o data/countmatrix_genus.h5 -db ~/.etetoolkit/taxa.sqlite
```

Selecting phylogeny.

```shell script
ONN select -i data/countmatrix_genus.h5 -phylo tmp/phylogeny_by_transformer.csv -o data/countmatrix_genus_selected.h5 -C 1e-3 -labels data/labels.h5 -dmax 5 -tmp tmp
```

Building ONN model from scratch and training.

```bash
ONN train -i data/matrix-genus-for-soil-C1e-3.h5 -label data/labels-for-soil.h5 -otlg config/ontology-for-soil.pkl -end-idx -1 -split-idx 10240 -log logs/training-history-for-soil.csv -dmax 6 -o ./models/model-for-soil -cfg ../../config/config.ini -phylo tmp/phylogeny_selected_using_varianceThreshold_C0.001.csv
```



## License

[![](https://award.dovolopor.com?lt=License&rt=MIT&rbc=green)](./LICENSE)