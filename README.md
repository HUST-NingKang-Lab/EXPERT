# EXPERT

### Exact and pervasive expert model for source tracking based on transfer learning

## Introduction

Based on Ontology-aware Neural Network architecture, EXPERT was trained using information-weighted loss to selectively learn from the training data, thus bypassing the negative impact of incomplete biome source information of data. Transfer learning technique was used to better adapt to novel biome ontology in a timely manner. 

The program is designed to help you to transfer Ontology-aware Neural Network model to other source tracking tasks. 

## Main features

- Fast, accurate and interpretable source tracking via **Ontology-aware Neural Network architecture**
- Selective learning from training data via **information-weighted loss**
- Fast adaptation to novel biome ontology via **transfer learning**
- Ultra-fast data cleaning via **in-memory NCBI taxonomy database**
- Parallelized feature encoding via `Tensorflow.keras`

## Installation

```shell script
python setup.py install
```

## Usage

#### Ontology construction

construct a biome ontology using "microbiomes.txt"

```shell script
expert construct -i tmp/microbiomes.txt -o ontology.pkl
```

#### Source mapping 

Mapping their source environments to microbiome ontology

```shell script
expert map -to-otlg -otlg tmp/ontology.pkl -i tmp/mapper.csv -o tmp/out/labels.h5 -unk
```

#### Data converting & cleaning

Convert input data to a count matrix in **genus** level

```shell script
expert convert -i tmp/sample-list.txt -tmp tmp -o data/countmatrix_genus.h5 -db ~/.etetoolkit/taxa.sqlite
```

#### Abundance mapping

map abundance data to phylogenetic tree and get abundance in **genus** level

```shell script
expert select -i data/countmatrix_genus.h5 -phylo tmp/phylogeny_by_transformer.csv -o data/countmatrix_genus_selected.h5 -C 1e-3 -labels data/labels.h5 -dmax 5 -tmp tmp
```

#### Ab initio training

Build EXPERT model from scratch and training

```bash
expert train -i data/matrix-genus-for-soil-C1e-3.h5 -label data/labels-for-soil.h5 -otlg config/ontology-for-soil.pkl -end-idx -1 -split-idx 10240 -log logs/training-history-for-soil.csv -dmax 6 -o ./models/model-for-soil -cfg ../../config/config.ini -phylo tmp/phylogeny_selected_using_varianceThreshold_C0.001.csv
```

#### Fast adaptation

```bash
expert transfer -model models/model-for-combied/ -i data/matrix-genus-for-disease-C1e-5.h5 -labels data/labels-for-disease.h5 -otlg config/ontology-for-disease.pkl -dmax 6 -o models/model-for-disease -cfg ../../config/config.ini  -phylo tmp/phylogeny_selected_using_varianceThreshold_C1e-05.csv -log logs/transfer_history.csv -tmp tmp -split-idx 10240 -end-idx -1
```

#### Source tracking

```bash
expert search -i data/matrix-genus-for-combined-C1e-5.h5 -model models/model-for-combied -cfg ../../config/config.ini -phylo tmp/phylogeny_selected_using_varianceThreshold_C1e-05_pog.csv -tmp tmp -o search_results -gpu -1
```

Run the program with `-h` option to see a detailed description on work modes & options.

## Input formats

EXPERT takes two kinds of **abundance data** and a **mapper file** (when transferring to other tasks) as inputs. 

#### Taxonomic assignments result for a single sample (OTU table)

Notice that here is a header "# Constructed from biom file" in the first line.

<table><thead><tr><th colspan="3"># Constructed from biom file</th></tr></thead><tbody><tr><td># OTU ID</td><td>ERR1754760</td><td>taxonomy</td></tr><tr><td>207119</td><td>19.0</td><td>sk__Archaea</td></tr><tr><td>118090</td><td>45.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__;o__Nitrosopumilales;f__Nitro...</td></tr><tr><td>153156</td><td>38.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__;o__Nitrosopumilales;f__Nitro...</td></tr><tr><td>131704</td><td>1.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__Nitrososphaeria;o__Nitrososp...</td></tr><tr><td>103181</td><td>5174.0</td><td>sk__Bacteria</td></tr><tr><td>157361</td><td>9.0</td><td>sk__Bacteria;k__;p__;c__;o__;f__;g__;s__agricultural_soil_bacterium_SC-I-11</td></tr></tbody></table>

#### Taxonomic assignments result for multiple samples (count matrix)

<table><thead><tr><th>#SampleID</th><th>ERR1844510</th><th>ERR1844449</th><th>ERR1844450</th><th>ERR1844451</th></tr></thead><tbody><tr><td>sk__Archaea</td><td>1.0</td><td>17.0</td><td>8.0</td><td>16.0</td></tr><tr><td>sk__Archaea;k__;p__Crenarchaeota</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>sk__Archaea;k__;p__Euryarchaeota</td><td>8.0</td><td>2.0</td><td>3.0</td><td>1.0</td></tr><tr><td>sk__Archaea;k__;p__Eury...;c__...</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>sk__Archaea;k__;p__Eury...;c__...;o__...</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></tbody></table>

#### Mapper file

<table><thead><tr><th></th><th>Env</th><th>SampleID</th></tr></thead><tbody><tr><td>0</td><td>root:Engineered:Wastewater</td><td>ERR2260442</td></tr><tr><td>1</td><td>root:Engineered:Wastewater</td><td>SRR980322</td></tr><tr><td>2</td><td>root:Engineered:Wastewater</td><td>ERR2985272</td></tr><tr><td>3</td><td>root:Engineered:Wastewater</td><td>ERR2814648</td></tr><tr><td>4</td><td>root:Engineered:Wastewater</td><td>ERR2985275</td></tr></tbody></table>

## Support

Feel free to contact us if you have any question. Thank you for using EXPERT.

## License

[![](https://award.dovolopor.com?lt=License&rt=MIT&rbc=green)](./LICENSE)