# EXPERT——一种用于量化源对微生物群落贡献的可扩展模型

在以快速、全面和上下文感知的方式对微生物组样本的来源进行量化方面仍然存在挑战。这种量化的传统方法在效率、准确性和可伸缩性之间不能权衡。

EXPERT，一个可扩展的社区级微生物溯源方法。基于生物群落本体信息和迁移学习技术，EXPERT能灵活地的感知上下文，可以很容易地扩展监督模型的搜索范围，包括上下文相关的群落样本和待研究的生物群落。同时，它在溯源的准确度和速度上都优于现有的方法。EXPERT的优势已经在多个溯源任务上得到了证明，包括纵向样本和在不同疾病阶段收集的溯源样本。详情请参考我们[最初研究](https://www.biorxiv.org/content/10.1101/2021.01.29.428751v1)。

高效率和准确性的监督学习遇到高可伸缩性的迁移学习，可以更好地理解微生物群落中的暗物质。



![迁移学习](https://github.com/HUST-NingKang-Lab/EXPERT/raw/master/docs/materials/transfer.png)



# 补充

如需使用专家，请[联系我们](https://github.com/HUST-NingKang-Lab/EXPERT#maintainer)。

这是我们的测试版，非常感谢您能给出任何评论或见解。



# 特征

1. 通过**迁移学习**适应微生物组研究的环境感知能力。
2. 通过**本体感知的正向传播**快速、准确和可解释的溯源。
3. 支持**扩增子测序**和**全基因组测序**数据。
4. 从部分标记的训练数据进行选择性学习。
5. 超快速的数据清洗，并且通过内置的NCBI分类数据库清洗数据。
6. 通过`tensorflow.keras`并行化编码



# 安装

您可以使用`pip`包管理器轻松地安装EXPERT。

```
pip install expert-mst        			# 安装EXPERT
expert init                   			# 初始化EXPERT并安装NCBI分类数据库
```



# 快速开始

通过一个案例帮助您快速了解EXPERT的基本功能，这已经在我们的预印[论文](https://doi.org/10.1101/2021.01.29.428751)中进行了相关介绍。我们还在另一个[存储库](https://github.com/HUST-NingKang-Lab/EXPERT-use-cases)中提供了功能更强的展示案例。

## 开始之前，您需要知道的事情

1. EXPERT的神奇之处是它可以对基本模型进行自动泛化，非深度学习的用户不需要任何编程技能，可以在终端上修改模型。我们概括了一个基本模型来监测结直肠癌(CRC)的进展，并评估该模型的性能。我们只使用训练过的疾病模型来量化带有不同疾病相关生物群系的宿主的贡献。(详情请参阅我们的预印本)
2. 微生物溯源：[贝叶斯全社区文化独立微生物溯源|自然方法](https://www.nature.com/articles/nmeth.1650)
3. 交叉验证：[交叉验证(统计)——维基百科](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

## 做好准备

请按照我们下面的介绍进行，确保这些命令都是在Linux或者Mac OSX平台上运行的。您还需要在开始前[安装 Anaconda](https://docs.anaconda.com/anaconda/install/)。

- 建议您安装expert-mst的0.2版本。

```
pip install https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2/expert-0.2_cpu-py3-none-any.whl
expert init
```

- 下载要使用的基本模型和数据集。`CM`是`countMatrix`的缩写，是丰度数据的一种格式(每一行代表一个分类单元，每一列代表一个样本)。`Mapper`是EXPERT的另一个重要输入，它记录作为输入样本的源生物群落。

```
wget -c https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2-m/disease_model.tgz
tar zxvf disease_model.tgz    			#解压基本模型。     
for file in {QueryCM.tsv,SourceCM.tsv,QueryMapper.csv,SourceMapper.csv}; do
		wget -c https://raw.githubusercontent.com/HUST-NingKang-Lab/EXPERT/master/data/$file;
done
```

## 预处理数据集

- 构建一个代表CRC阶段的生物群系本体。在打印显示的界面中，您将看到像树一样的构造本体。

```
grep -v "Env" SourceMapper.csv | awk -F ',' '{print $6}' | sort | uniq > microbiomes.txt
expert construct -i microbiomes.txt -o ontology.pkl
```

- 将微生物群落样本映射到生物群落本体上，获得分级标签。您将在打印的界面中看到每个生物群落本体层上的样本数目。

```
expert map --to-otlg -i SourceMapper.csv -t ontology.pkl -o SourceLabels.h5
expert map --to-otlg -i QueryMapper.csv -t ontology.pkl -o QueryLabels.h5
```

- 处理作为输入的丰度数据，将其处理为模型可识别的`hdf`文件。EXPERT只接受标准化的丰度数据，我们使用`convert`模型对丰度数据进行标准化。

```
ls SourceCM.tsv > inputList; expert convert -i inputList -o SourceCM.h5 --in-cm
ls QueryCM.tsv > inputList; expert convert -i inputList -o QueryCM.h5 --in-cm
rm inputList
```

## 建模和评估

- 将疾病知识(从疾病模型)转移到CRC模型，以便更好地进行CRC监测。您将在打印的界面中看到运行日志和训练过程。

```
expert transfer -i SourceCM.h5 -l SourceLabels.h5 -t ontology.pkl -m disease_model -o CRC_model
```

- 根据模型搜索测试样本。

```
expert search -i QueryCM.h5 -m CRC_model -o quantified_source_contributions
```

- 评估CRC模型的性能。您将获得一份CRC各个阶段的情况报告。

```
expert evaluate -i quantified_source_contributions -l QueryLabels.h5 -o performance_report
cat performance_report/overall.csv
```

您现在已经获得了微生物溯源的EXPERT建模技能。接下来，您可能想探讨一个问题：哪个基本模型在CRC监控中表现最好呢？您可能想要使用另一个基本模型来评估性能，祝您好运！



# 高级用法

EXPERT已经使适应了依赖于上下文的研究，在这些研究中您可以选择潜在的资源进行估计。请参阅我们的文件：[高级用法](https://github.com/HUST-NingKang-Lab/EXPERT/wiki/advanced-usage)。



# 模型来源

|   模型   |                   生物群落本体                   |  顶级生物群落  |                   数据来源                    | 数据集大小 |                           下载链接                           |             备注             |
| :------: | :----------------------------------------------: | :------------: | :-------------------------------------------: | :--------: | :----------------------------------------------------------: | :--------------------------: |
| 基本模型 | 地球上132个生物群落的生物群落本体(截至2020年1月) |      root      | [MGnify](https://www.ebi.ac.uk/metagenomics/) |  115,892   | [下载](https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2-m/general_model.tgz) | MGnify对样品的处理**不**均匀 |
| 人类模型 |        27个人类相关生物群系的生物群系本体        |     human      | [MGnify](https://www.ebi.ac.uk/metagenomics/) |   52,537   | [下载](https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2-m/human_model.tgz) | MGnify对样品的处理**不**均匀 |
| 疾病模型 |      20个人类疾病相关生物群系的生物群系本体      | root(人类肠道) |  [GMrepo](https://gmrepo.humangut.info/home)  |   13,642   | [下载](https://github.com/HUST-NingKang-Lab/EXPERT/releases/download/v0.2-m/disease_model.tgz) |    MGnify对样品的处理均匀    |

备注：这些模型是在EXPERT的0.2版本上进行训练。



# 如何引用

如果您在相关出版物中使用EXPERT(或受到该方法的启发)，我们希望引用以下论文：

```
Enabling technology for microbial source tracking based on transfer learning: From ontology-aware general knowledge to context-aware expert systems
Hui Chong, Qingyang Yu, Yuguo Zha, Guangzhou Xiong, Nan Wang, Chuqing Sun, Sicheng Wu, Weihua Chen, Kang Ning
bioRxiv 2021.01.29.428751; doi: https://doi.org/10.1101/2021.01.29.428751
```



# 联系方式

| 姓名 |                       电子邮件                        |                所属机构                |
| :--: | :---------------------------------------------------: | :------------------------------------: |
| 冲辉 | [huichong.me@gmail.com](mailto:huichong.me@gmail.com) | 华中科技大学生命科学与技术学院研究助理 |
|黄士娟| [hshijuan@qq.com](mailto:hshijuan@qq.com)             | 华中科技大学生命科学与技术学院本科生   |
| 宁康 |  [ningkang@hust.edu.cn](mailto:ningkang@hust.edu.cn)  |   华中科技大学生命科学与技术学院教授   |



