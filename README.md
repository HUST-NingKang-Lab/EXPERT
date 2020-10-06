# Generalized Ontology-aware Neural Network

Generalized ontology-aware neural network for ontological data mining.

Using number of sample > 50 sources data.**

## Deployment

```shell script
pip install GONN
```

## Usage

## Demo

Change working directory.

```shell script
cd others
```

Getting a list of input files.

```shell script
ls ../data/*/*  > tmp/sample-list.txt
```

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

## Reference

- [如何从模板创建仓库？](https://help.github.com/cn/articles/creating-a-repository-from-a-template)
- [如何发布自己的包到 pypi ？](https://www.v2ai.cn/python/2018/07/30/PY-1.html)
