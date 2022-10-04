## EXPERT - a scalable model for quantifying source contributions for microbial communities

[![](https://img.shields.io/badge/PyPI-v0.3-blue?style=flat)](https://pypi.org/project/expert-mst/) 
[![Downloads](https://pepy.tech/badge/expert-mst)](https://pepy.tech/project/expert-mst)
![](https://img.shields.io/badge/status-beta-yellow?style=flat) 
[![](https://img.shields.io/badge/DOI-10.1093/bib/bbac396-brightgreen?style=flat)](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac396/6702669) 
![](https://img.shields.io/github/license/HUST-NingKang-Lab/EXPERT?style=flat) 
[![](https://img.shields.io/badge/README-CN-green?style=flat)](https://github.com/HUST-NingKang-Lab/EXPERT/blob/master/README_cn.md) 

Challenges remain to be addressed in terms of quantifying source origins for microbiome samples in a fast, comprehensive, and context-aware manner. Traditional approaches to such quantification have severe trade-offs between efficiency, accuracy, and scalability. 

Here, we introduce EXPERT, a scalable community-level microbial source tracking approach. Built upon the biome ontology information and transfer learning techniques, EXPERT has acquired the context-aware flexibility and could easily expand the supervised model's search scope to include the context-depende/nt community samples and understudied biomes. While at the same time, it is superior to current approaches in source tracking accuracy and speed. EXPERT's superiority has been demonstrated on multiple source tracking tasks, including source tracking samples collected at different disease stages and longitudinal samples. For details refer to our [original study](https://www.biorxiv.org/content/10.1101/2021.01.29.428751v1). 

Supervised learning (with high efficiency and accuracy) meets transfer learning (with inherent high scalability), towards better understanding the dark matters in microbial community.

<img src="https://github.com/HUST-NingKang-Lab/EXPERT/raw/master/docs/materials/transfer.png" style="zoom:150%;" />

## Support

For support using EXPERT, please [contact us](https://github.com/HUST-NingKang-Lab/EXPERT#maintainer). 

This is our beta version, any comments or insights would be greatly appreciated. 

## Features

1. Context-aware ability to adapt to microbiome studies via **transfer learning**
2. Fast, accurate and interpretable source tracking via **ontology-aware forward propagation**
3. Supports both **amplicon sequencing** and **whole-genome sequencing** data. 
4. Selective learning from partially-labeled training data
5. Ultra-fast data cleaning & cleaning via in-memory NCBI taxonomy database
6. Parallelized feature encoding via `tensorflow.keras`

## Installation

You can simply install EXPERT using `pip` package manager.

```bash
pip install expert-mst    # Install EXPERT
expert init               # Initialize EXPERT and install NCBI taxonomy database
```

## Quick start

Here we quickly go-through basic functionalities of EXPERT through a case study, which have already been conducted in our preprinted [paper](https://doi.org/10.1101/2021.01.29.428751). We also provided more functional show-cases in another [repository](https://github.com/HUST-NingKang-Lab/EXPERT-use-cases). 

#### Things to know before starting

1. EXPERT's fantastic function is its automatic generalization of fundamental models, which allows non-deep-learning users to modify the models just in terminal, without the need of any programming skill. Here we generalize a fundamental model for monitoring the progression of colorectal cancer (CRC) and assess the performance of the generalized model. We only use the disease model trained for quantifying contribution from hosts with different disease-associated biomes (refer to our preprint for details).

2. Microbial source tracking: [Bayesian community-wide culture-independent microbial source tracking | Nature Methods](https://www.nature.com/articles/nmeth.1650)

3. Cross-validation: [Cross-validation (statistics) - Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

#### Get prepared

Please follow our instructions below and make sure all these commands were run on Linux/Mac OSX platform. You may also need to [install Anaconda](https://docs.anaconda.com/anaconda/install/) before starting. 

- Install expert-mst version 0.2 (suggested).

```bash
pip install https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2/expert-0.2_cpu-py3-none-any.whl
expert init
```

- Download the fundamental model and dataset to be used. Here `CM` is a abbreviation term of `countMatrix`, which is a format of abundance data (each row represents a taxon, and each column represents a sample/run). `Mapper` is another important input of EXPERT, which records source biomes for input samples.

```bash
wget -c https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2-m/disease_model.tgz
tar zxvf disease_model.tgz  # Decompress the fundamental model.
for file in {QueryCM.tsv,SourceCM.tsv,QueryMapper.csv,SourceMapper.csv}; do 
	wget -c https://raw.githubusercontent.com/HUST-NingKang-Lab/EXPERT/master/data/$file;
done
```

#### Preprocess the dataset

- Construct a biome ontology representing stages of CRC. You'll see constructed ontology like a tree in the printed message.

```bash
grep -v "Env" SourceMapper.csv | awk -F ',' '{print $6}' | sort | uniq > microbiomes.txt
expert construct -i microbiomes.txt -o ontology.pkl
```

- Map microbial community samples to the biome ontology to obtain hierarchical labels. You'll see counts of the samples on each biome ontology layer in the printed message.

```bash
expert map --to-otlg -i SourceMapper.csv -t ontology.pkl -o SourceLabels.h5
expert map --to-otlg -i QueryMapper.csv -t ontology.pkl -o QueryLabels.h5
```

- Convert input abundance data to model-acceptable `hdf` file. The EXPERT model only accepts standardized abundance data. Here we standardize the abundance data using `convert` mode.

```bash
ls SourceCM.tsv > inputList; expert convert -i inputList -o SourceCM.h5 --in-cm;
ls QueryCM.tsv > inputList; expert convert -i inputList -o QueryCM.h5 --in-cm;
rm inputList
```

#### Modeling and evaluation

- Transfer knowledge about disease (from the disease model) to the CRC model, for a better performance on the CRC monitoring. You'll see running log and training process in the printed message.

```bash
expert transfer -i SourceCM.h5 -l SourceLabels.h5 -t ontology.pkl -m disease_model -o CRC_model
```

- Search the query samples against the model. 

```bash
expert search -i QueryCM.h5 -m CRC_model -o quantified_source_contributions
```

- Evaluate the performance of the CRC model. You'll obtain a performance report on each stage of CRC.

```bash
expert evaluate -i quantified_source_contributions -l QueryLabels.h5 -o performance_report
cat performance_report/overall.csv
```

You now have acquired skills of EXPERT modeling for microbial source tracking. Next, you may want to explore a question: Which fundamental model gives the best performance on the CRC monitoring? You may want to assess the performance utilizing another fundamental model. Good luck.

## Advanced usage

EXPERT has enabled the adaptation to context-dependent studies, in which you can choose potential sources to be estimated. Please follow our [documentation: advanced usage](https://github.com/HUST-NingKang-Lab/EXPERT/wiki/advanced-usage).

## Model resources

| Model         | Biome ontology                                           | Top-level biome  | Data source                                   | Dataset size | Download link                                                | Note                                                   |
| ------------- | -------------------------------------------------------- | ---------------- | --------------------------------------------- | ------------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| general model | biome ontology for 132 biomes on earth (as of Jan. 2020) | root             | [MGnify](https://www.ebi.ac.uk/metagenomics/) | 115,892      | [download](https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2-m/general_model.tgz) | The samples were **not** uniformly processed by MGnify |
| human model   | biome ontology for 27 human-associated biomes            | human            | [MGnify](https://www.ebi.ac.uk/metagenomics/) | 52,537       | [download](https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2-m/human_model.tgz) | The samples were **not** uniformly processed by MGnify |
| disease model | biome ontology for 20 human disease-associated biomes    | root (human gut) | [GMrepo](https://gmrepo.humangut.info/home)   | 13,642       | [download](https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2-m/disease_model.tgz) | The samples were uniformly processed by GMrepo         |

Note: These models were trained on EXPERT version 0.2.

## How-to-cite

If you are using EXPERT in a scientific publication (or inspired by the approach), we would appreciate citations to the following paper:

> Hui Chong, Yuguo Zha, Qingyang Yu, Mingyue Cheng, Guangzhou Xiong, Nan Wang, Xinhe Huang, Shijuan Huang, Chuqing Sun, Sicheng Wu, Wei-Hua Chen, Luis Pedro Coelho, Kang Ning*. EXPERT: transfer learning-enabled context-aware microbial community classification. Briefings in Bioinformatics, 2022; bbac396. doi:[10.1093/bib/bbac396](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac396/6702669).

## Maintainer

|   Name    | Email                 | Organization                                                 |
| :-------: | --------------------- | ------------------------------------------------------------ |
| Hui Chong | huichong.me@gmail.com | Research Assistant, School of Life Science and Technology, Huazhong University of Science & Technology |
| Xinhe Huang | huangxinhe@hust.edu.cn | Undergraduate,School of Life Science and Technology, Huazhong University of Science & Technology|
| Shijuan Huang | hshijuan@qq.com      | Undergraduate,School of Life Science and Technology, Huazhong University of Science & Technology|
| Kang Ning | ningkang@hust.edu.cn  | Professor, School of Life Science and Technology, Huazhong University of Science & Technology |
