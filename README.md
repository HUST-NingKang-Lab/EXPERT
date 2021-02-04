# EXPERT - a scalable model for quantifying source contributions for microbial communities

[![](https://img.shields.io/badge/PYPI-v0.3-blue?style=flat-square&logo=appveyor)](https://pypi.org/project/expert-mst/) ![](https://img.shields.io/badge/status-beta-yellow?style=flat-square&logo=appveyor) [![](https://img.shields.io/badge/DOI-10.1101/2021.01.29.428751-brightgreen?style=flat-square&logo=appveyor)](https://www.biorxiv.org/content/10.1101/2021.01.29.428751v1) ![](https://img.shields.io/github/license/HUST-NingKang-Lab/EXPERT?style=flat-square&logo=appveyor) [![](https://img.shields.io/badge/support-huichong.me@gmail.com-blue?style=flat-square&logo=appveyor)](mailto:huichong.me@gmail.com)

Challenges remain to be addressed in terms of quantifying source origins for microbiome samples in a fast, comprehensive, and context-aware manner. Traditional approaches to such quantification have severe trade-offs between efficiency, accuracy, and scalability. 

Here, we introduce EXPERT, a scalable community-level microbial source tracking approach. Built upon the biome ontology information and transfer learning techniques, EXPERT has acquired the context-aware flexibility and could easily expand the supervised model's search scope to include the context-depende/nt community samples and understudied biomes. While at the same time, it is superior to current approaches in source tracking accuracy and speed. EXPERT's superiority has been demonstrated on multiple source tracking tasks, including source tracking samples collected at different disease stages and longitudinal samples. For details refer to our [original study](https://www.biorxiv.org/content/10.1101/2021.01.29.428751v1). 

Supervised learning (with high efficiency and accuracy) meets transfer learning (with inherent high scalability), towards better understanding the dark matters in microbial community.

## Support

For support using EXPERT, please [contact us](https://github.com/HUST-NingKang-Lab/EXPERT#maintainer). 

This is our beta version, any comments or insights would be greatly appreciated. 

## Current features

- Context-aware ability to adapt to microbiome studies via **transfer learning**
- Fast, accurate and interpretable source tracking via **ontology-aware forward propagation**
- Supports both **amplicon sequencing** and **whole-genome sequencing** data. 
- Selective learning from partially-labeled training data
- Ultra-fast data cleaning & cleaning via in-memory NCBI taxonomy database
- Parallelized feature encoding via `tensorflow.keras`

## Installation

You can simply install EXPERT using `pip` package manager.

```bash
pip install expert-mst    # Install EXPERT
expert init               # Initialize EXPERT and install NCBI taxonomy database
```

## Quick start

Convert input abundance data to model-acceptable ` hdf` file.

```bash
expert convert -i countMatrices.txt -o countMatrix.h5 --in-cm
```

Source track microbial communities. Here you can specify an EXPERT model for the source tracking.  We have provided our general model, human model and disease model (refer to our [original study](https://www.biorxiv.org/content/10.1101/2021.01.29.428751v1) for details).

```bash
expert search -i countMatrix.h5 -o searchResult -m model
```

## Advanced usage

EXPERT has enabled the adaptation to context-dependent studies, in which you can choose potential sources to be estimated. Please follow our [documentation: advanced usage](https://github.com/HUST-NingKang-Lab/EXPERT/wiki/advanced-usage).

More functional show-cases can be found at [EXPERT-use-cases](https://github.com/HUST-NingKang-Lab/EXPERT-use-cases). 

## How-to-cite

If you are using EXPERT in a scientific publication (or inspired by the approach), we would appreciate citations to the following paper:

```
Enabling technology for microbial source tracking based on transfer learning: From ontology-aware general knowledge to context-aware expert systems
Hui Chong, Qingyang Yu, Yuguo Zha, Guangzhou Xiong, Nan Wang, Chuqing Sun, Sicheng Wu, Weihua Chen, Kang Ning
bioRxiv 2021.01.29.428751; doi: https://doi.org/10.1101/2021.01.29.428751
```

## Maintainer

|   Name    | Email                 | Organization                                                 |
| :-------: | --------------------- | ------------------------------------------------------------ |
| Hui Chong | huichong.me@gmail.com | Research Assistant, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kang Ning | ningkang@hust.edu.cn  | Professor, School of Life Science and Technology, Huazhong University of Science & Technology |