# Generalized Ontology-aware Neural Network

Generalized ontology-aware neural network for ontological data mining.

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
ONN convert -gen-phylo -i tmp/sample-list.txt -conf tmp/conf -o tmp/out/countmatrix_genus.h5 -db ~/.etetoolkit/taxa.sqlite
```

Selecting top 1000 important phylogeny.

```shell script
ONN select -i tmp/conf/phylogeny_by_transformer.csv -cm tmp/out/countmatrix_genus.h5 -o tmp/conf/phylogeny_top1000.csv -top 1000 -labels tmp/out/labels.h5 -dmax 5
```

Converting input data to count matrix at each rank in [sk, p, c, o, f, g].


```shell script
ONN convert -i tmp/sample-list.txt -conf tmp/conf -o tmp/out/countmatrix_each_rank.h5 -db ~/.etetoolkit/taxa.sqlite -phylo tmp/conf/phylogeny_top1000.csv
```

## License

[![](https://award.dovolopor.com?lt=License&rt=MIT&rbc=green)](./LICENSE)

## Reference

- [如何从模板创建仓库？](https://help.github.com/cn/articles/creating-a-repository-from-a-template)
- [如何发布自己的包到 pypi ？](https://www.v2ai.cn/python/2018/07/30/PY-1.html)
