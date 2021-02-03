# EXPERT - a scalable model for quantifying source contributions for microbial communities

Challenges remain to be addressed in terms of quantifying source origins for microbiome samples in an ultrafast, comprehensive, and context-aware manner. Traditional approaches to such quantification have severe trade-offs between efficiency, accuracy, and scalability. 

Here, we introduce EXPERT, a scalable community-level microbial source tracking approach. Built upon the biome ontology information and transfer learning techniques, EXPERT has acquired the context-aware flexibility and could easily expand the supervised model's search scope to include the context-dependent community samples and understudied biomes. While at the same time, it is superior to current approaches in source tracking accuracy and speed. EXPERT's superiority has been demonstrated on multiple source tracking tasks, including source tracking samples collected at different disease stages and longitudinal samples. For details refer to our [original study]([Enabling technology for microbial source tracking based on transfer learning: From ontology-aware general knowledge to context-aware expert systems | bioRxiv](https://www.biorxiv.org/content/10.1101/2021.01.29.428751v1)). 

Supervised learning (with high efficiency and accuracy) meets transfer learning (with inherent high scalability), towards better understanding the dark matters in microbial community.

If you use EXPERT in a scientific publication, we would appreciate citations to the following paper:

```
Enabling technology for microbial source tracking based on transfer learning: From ontology-aware general knowledge to context-aware expert systems
Hui Chong, Qingyang Yu, Yuguo Zha, Guangzhou Xiong, Nan Wang, Chuqing Sun, Sicheng Wu, Weihua Chen, Kang Ning
bioRxiv 2021.01.29.428751; doi: https://doi.org/10.1101/2021.01.29.428751
```

This is our beta version, any comments or insights would be greatly appreciated. 

For support using EXPERT, feel free to [contact us](https://github.com/HUST-NingKang-Lab/EXPERT#maintainer). Thank you for using EXPERT.

## Current features

- Context-aware ability to adapt to microbiome studies via **transfer learning**
- Fast, accurate and interpretable source tracking via **ontology-aware forward propagation**
- Supports both **amplicon sequencing** and **whole-genome sequencing** data. 
- Selective learning from training data
- Ultra-fast data cleaning & cleaning via in-memory NCBI taxonomy database
- Parallelized feature encoding via `tensorflow.keras`

## Installation

- Install EXPERT through `pip` package manager

```shell script
pip install expert-mst
```

- Initialize EXPERT and install NCBI taxonomy database

```
expert init
```

## Quick start

Convert input abundance data to model-acceptable ` hdf` file.

```
expert convert -i countMatrices.txt -o countMatrix.h5 --in-cm
```

Source track microbial communities. Here you can specify an EXPERT model for the source tracking.  We have provided our general model, human model and disease model (refer to our [original study](https://www.biorxiv.org/content/10.1101/2021.01.29.428751v1) for details).

```
expert search -i countMatrix.h5 -o searchResult -m model
```

## Advanced usage

EXPERT has enabled the adaptation to context-dependent studies, in which you can choose potential sources to be estimated. Please follow our [documentation: advanced usage](https://github.com/HUST-NingKang-Lab/EXPERT/wiki/advanced-usage).

## Input abundance data

EXPERT takes two kinds of **abundance data** as inputs. 

1. Taxonomic assignments result for a single sample (abundance table). The example shown below is obtained through amplicon sequencing. Note that here is a header "# Constructed from biom file" in the first line.

<table><thead><tr><th colspan="3"># Constructed from biom file</th></tr></thead><tbody><tr><td># OTU ID</td><td>ERR1754760</td><td>taxonomy</td></tr><tr><td>207119</td><td>19.0</td><td>sk__Archaea</td></tr><tr><td>118090</td><td>45.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__;o__Nitrosopumilales;f__Nitro...</td></tr><tr><td>153156</td><td>38.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__;o__Nitrosopumilales;f__Nitro...</td></tr><tr><td>131704</td><td>1.0</td><td>sk__Archaea;k__;p__Thaumarchaeota;c__Nitrososphaeria;o__Nitrososp...</td></tr><tr><td>103181</td><td>5174.0</td><td>sk__Bacteria</td></tr><tr><td>157361</td><td>9.0</td><td>sk__Bacteria;k__;p__;c__;o__;f__;g__;s__agricultural_soil_bacterium_SC-I-11</td></tr></tbody></table>

2. Taxonomic assignments result for multiple samples (count matrix)

<table><thead><tr><th>#SampleID</th><th>ERR1844510</th><th>ERR1844449</th><th>ERR1844450</th><th>ERR1844451</th></tr></thead><tbody><tr><td>sk__Archaea</td><td>1.0</td><td>17.0</td><td>8.0</td><td>16.0</td></tr><tr><td>sk__Archaea;k__;p__Crenarchaeota</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>sk__Archaea;k__;p__Euryarchaeota</td><td>8.0</td><td>2.0</td><td>3.0</td><td>1.0</td></tr><tr><td>sk__Archaea;k__;p__Eury...;c__...</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>sk__Archaea;k__;p__Eury...;c__...;o__...</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></tbody></table>

## License

[![](https://award.dovolopor.com?lt=License&rt=MIT&rbc=green)](./LICENSE)

## Maintainer

|   Name    | Email                 | Organization                                                 |
| :-------: | --------------------- | ------------------------------------------------------------ |
| Hui Chong | huichong.me@gmail.com | Research Assistant, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kang Ning | ningkang@hust.edu.cn  | Professor, School of Life Science and Technology, Huazhong University of Science & Technology |