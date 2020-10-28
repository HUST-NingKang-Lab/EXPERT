# EXPERT

### Exact and pervasive expert model for source tracking based on transfer learning

## Introduction

Based on Ontology-aware Neural Network architecture, EXPERT was trained using information-weighted loss to selectively learn from the training data, thus bypassing the negative impact of incomplete biome source information of data. Transfer learning technique was used to better adapt to novel biome ontology in a timely manner. 

The program is designed to help you to transfer Ontology-aware Neural Network model to other source tracking tasks. Feel free to contact us if you have any question. Thank you for using EXPERT.

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

construct a biome ontology using `microbiomes.txt`

```shell script
expert construct -i microbiomes.txt -o ontology.pkl
```

- Input: `microbiomes.txt` file, contains path from "root" node to each leaf node of biome ontology.

```
root:Environmental:Terrestrial:Soil
root:Environmental:Terrestrial:Soil:Agricultural
root:Environmental:Terrestrial:Soil:Boreal_forest
root:Environmental:Terrestrial:Soil:Contaminated
root:Environmental:Terrestrial:Soil:Crop
root:Environmental:Terrestrial:Soil:Crop:Agricultural_land
root:Environmental:Terrestrial:Soil:Desert
root:Environmental:Terrestrial:Soil:Forest_soil
root:Environmental:Terrestrial:Soil:Grasslands
root:Environmental:Terrestrial:Soil:Loam:Agricultural
root:Environmental:Terrestrial:Soil:Permafrost
root:Environmental:Terrestrial:Soil:Sand
root:Environmental:Terrestrial:Soil:Tropical_rainforest
root:Environmental:Terrestrial:Soil:Uranium_contaminated
root:Environmental:Terrestrial:Soil:Wetlands
root:Host-associated:Plants:Rhizosphere:Soil
```

- Output: constructed biome ontology (pickle format, non-human-readable).

#### Source mapping 

Mapping their source environments to microbiome ontology

```shell script
expert map -to-otlg -otlg ontology.pkl -i mapper.csv -o labels.h5
```

- Input: the mapper file, contains biome source information for samples.

<table><thead><tr><th></th><th>Env</th><th>SampleID</th></tr></thead><tbody><tr><td>0</td><td>root:Engineered:Wastewater</td><td>ERR2260442</td></tr><tr><td>1</td><td>root:Engineered:Wastewater</td><td>SRR980322</td></tr><tr><td>2</td><td>root:Engineered:Wastewater</td><td>ERR2985272</td></tr><tr><td>3</td><td>root:Engineered:Wastewater</td><td>ERR2814648</td></tr><tr><td>4</td><td>root:Engineered:Wastewater</td><td>ERR2985275</td></tr></tbody></table>

- Output: the labels for samples in each layer of the biome ontology (HDF format, non-human-readable).

#### Data converting & cleaning

Convert input data to a count matrix in **genus** level.

```shell script
expert convert -i countMatrices.txt -in-cm -o countMatrix.h5
```

- Input: a text file contains path to input count matrix file / OTU table.

```
datasets/soil_dataset/root:Host-associated:Plants:Rhizosphere:Soil/MGYS00005146-ERR1690680.tsv
datasets/soil_dataset/root:Host-associated:Plants:Rhizosphere:Soil/MGYS00005146-ERR1689675.tsv
datasets/soil_dataset/root:Host-associated:Plants:Rhizosphere:Soil/MGYS00000513-ERR986792.tsv
datasets/soil_dataset/root:Host-associated:Plants:Rhizosphere:Soil/MGYS00005146-ERR1691198.tsv
datasets/soil_dataset/root:Host-associated:Plants:Rhizosphere:Soil/MGYS00001704-ERR1905845.tsv
datasets/soil_dataset/root:Host-associated:Plants:Rhizosphere:Soil/MGYS00005146-ERR1689214.tsv
datasets/soil_dataset/root:Host-associated:Plants:Rhizosphere:Soil/MGYS00005146-ERR1689910.tsv
```

- Output: converted count matrix file in **genus** level (HDF format, non-human-readable). 

#### Ab initio training

Build EXPERT model from scratch and training

```bash
expert train -i countMatrix.h5 -labels labels.h5 -otlg ontology.pkl -o model -log history.csv
```

- Input: biome ontology and converted count matrix in genus level (and also labels for samples involved in the count matrix).
- Output: trained model.

#### Fast adaptation

```bash
expert transfer -i countMatrix.h5 -labels labels.h5 -otlg ontology.pkl -o model -log history.csv
```

- Input: biome ontology and converted count matrix in genus level (and also labels for samples involved in the count matrix).
- Output: trained model.

#### Source tracking

```bash
expert search -i countMatrix.h5 -o searchResult -gpu -1
```

- Input: converted count matrix in genus level.
- Output: search result (multi-layer ).

```
├── layer-2.csv
├── layer-3.csv
├── layer-4.csv
├── layer-5.csv
└── layer-6.csv
```

Take `layer-2.csv` as an example.

|            | root:Engineered | root:Environmental | root:Host-associated | root:Mixed    | Unknown     |
| ---------- | --------------- | ------------------ | -------------------- | ------------- | ----------- |
| ERR2278752 | 0.0041427016    | 0.26372418         | 0.68632126           | 0.00040003657 | 0.045411825 |
| ERR2278753 | 0.002841179     | 0.07928896         | 0.91735524           | 0.00051463145 | 0.0         |
| ERR2666855 | 0.0006751048    | 0.0021803565       | 0.9970531            | 9.1493675e-05 | 0.0         |
| ERR2666860 | 0.0005227786    | 0.013902989        | 0.98542625           | 0.00014803928 | 0.0         |
| ERR2666881 | 0.0009569057    | 0.0023957777       | 0.9965403            | 0.00010694566 | 0.0         |

#### Evaluation

```bash
expert evaluate -i searchResultFolder -labels labels.h5 -T 100 -o EvaluationReport -p NUMProcesses
```

- Input: multi-layer labels and search result (source contribution) for samples.
- Output: label-based evaluation report.

```
├── layer-2
│   └── root:Host-associated.csv
├── layer-2.csv
├── layer-3
│   └── root:Host-associated:Human.csv
├── layer-3.csv
├── layer-4
│   ├── root:Host-associated:Human:Circulatory_system.csv 
│   ├── root:Host-associated:Human:Digestive_system.csv
│   ├── root:Host-associated:Human:Lympathic_system.csv
│   ├── root:Host-associated:Human:Reproductive_system.csv
│   ├── root:Host-associated:Human:Respiratory_system.csv
│   └── root:Host-associated:Human:Skin.csv
├── layer-4.csv
├── layer-5
│   ├── root:Host-associated:Human:Circulatory_system:Blood.csv
│   ├── ...
│   └── root:Host-associated:Human:Respiratory_system:Pulmonary_system.csv
├── layer-5.csv
├── layer-6
│   ├── root:Host-associated:Human:Digestive_system:Large_intestine:Fecal.csv
│   ├── ...
│   └── root:Host-associated:Human:Respiratory_system:Pulmonary_system:Sputum.csv
└── layer-6.csv
```

Take `layer-4/root:Host-associated:Human:Skin.csv` as an example.

| t    | TN    | FP    | FN   | TP   | Acc    | Sn     | Sp     | TPR    | FPR    | Rc     | Pr     | F1     | ROC-AUC | F-max  |
| ---- | ----- | ----- | ---- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- | ------ |
| 0.0  | 0     | 47688 | 0    | 4847 | 0.0923 | 1.0    | 0.0    | 1.0    | 1.0    | 1.0    | 0.0923 | 0.1689 | 0.9951  | 0.9374 |
| 0.01 | 44794 | 2893  | 30   | 4816 | 0.9444 | 0.9938 | 0.9393 | 0.9938 | 0.0607 | 0.9938 | 0.6247 | 0.7672 | 0.9951  | 0.9374 |
| 0.02 | 45545 | 2142  | 44   | 4802 | 0.9584 | 0.9909 | 0.9551 | 0.9909 | 0.0449 | 0.9909 | 0.6915 | 0.8146 | 0.9951  | 0.9374 |
| 0.03 | 45934 | 1753  | 59   | 4787 | 0.9655 | 0.9878 | 0.9632 | 0.9878 | 0.0368 | 0.9878 | 0.732  | 0.8409 | 0.9951  | 0.9374 |
| 0.04 | 46228 | 1459  | 73   | 4773 | 0.9708 | 0.9849 | 0.9694 | 0.9849 | 0.0306 | 0.9849 | 0.7659 | 0.8617 | 0.9951  | 0.9374 |

Run the program with `-h` option to see a detailed description on work modes & options.

## Input abundance data

EXPERT takes two kinds of **abundance data **as inputs. 

#### Taxonomic assignments result for a single sample (OTU table)

Notice that here is a header "# Constructed from biom file" in the first line.

<table><thead><tr><th colspan="3"># Constructed from biom file</th></tr></thead><tbody><tr><td># OTU ID</td><td>ERR1754760</td><td>taxonomy</td></tr><tr><td>207119</td><td>19.0</td><td>sk__Archaea</td></tr><tr><td>118090</td><td>45.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__;o__Nitrosopumilales;f__Nitro...</td></tr><tr><td>153156</td><td>38.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__;o__Nitrosopumilales;f__Nitro...</td></tr><tr><td>131704</td><td>1.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__Nitrososphaeria;o__Nitrososp...</td></tr><tr><td>103181</td><td>5174.0</td><td>sk__Bacteria</td></tr><tr><td>157361</td><td>9.0</td><td>sk__Bacteria;k__;p__;c__;o__;f__;g__;s__agricultural_soil_bacterium_SC-I-11</td></tr></tbody></table>

#### Taxonomic assignments result for multiple samples (count matrix)

<table><thead><tr><th>#SampleID</th><th>ERR1844510</th><th>ERR1844449</th><th>ERR1844450</th><th>ERR1844451</th></tr></thead><tbody><tr><td>sk__Archaea</td><td>1.0</td><td>17.0</td><td>8.0</td><td>16.0</td></tr><tr><td>sk__Archaea;k__;p__Crenarchaeota</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>sk__Archaea;k__;p__Euryarchaeota</td><td>8.0</td><td>2.0</td><td>3.0</td><td>1.0</td></tr><tr><td>sk__Archaea;k__;p__Eury...;c__...</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>sk__Archaea;k__;p__Eury...;c__...;o__...</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></tbody></table>

## License

[![](https://award.dovolopor.com?lt=License&rt=MIT&rbc=green)](./LICENSE)

## Maintainer

|   Name    | Email                 | Organization                                                 |
| :-------: | --------------------- | ------------------------------------------------------------ |
| Hui Chong | huichong.me@gmail.com | Research Assistant, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kang Ning | ningkang@hust.edu.cn  | Professor, School of Life Science and Technology, Huazhong University of Science & Technology |