## 环境：
	python=3.7
    pytorch==1.12   


## 模板提取：
* 运行脚本的的命令：
    * cd LogAnomaly/ft_tree/
    * python -u ft_tree.py -data_path ../data/kafka.csv -template_path ../middle/kafka_log.template -fre_word_path ../middle/kafka_log.fre
* 参数样例：
    * data_path：日志文件 ../data/kafka.csv
    * template_path：模板文件 ../middle/kafka_log.template
    * fre_word_path：单词词频文件 ../middle/kafka_log.fre

## 模板匹配：
* 运行脚本的的命令：
    * cd LogAnomaly/ft_tree/
    * python -u matchTemplate.py -template_path ../middle/kafka_log.template -fre_word_path ../middle/kafka_log.fre -log_path ../data/kafka.csv -out_seq_path ../data/kafka_log.seq -match_model 1
* 参数样例：
    * template_path：模板文件 ../middle/kafka_log.template
    * fre_word_path：单词词频文件 ../middle/kafka_log.fre
    * log_path：日志文件 ../data/kafka.csv
    * out_seq_path：日志序列文件 ../data/kafka_log.seq
    * match_model：1:正常匹配日志  2:单条增量学习&匹配 3:批量增量学习&匹配

## 获取正常日志序列：
* 运行脚本的的命令：
    * cd LogAnomaly/data/
    * python removeAnomaly.py -input_seq kafka_log.seq -input_label kafka_log.label
* 参数样例：
    * input_seq：日志序列文件 kafka_log.seq
    * input_label：日志label文件 kafka_log.label


## 基于模板搜索同义词和反义词：
* 运行脚本的的命令：
    * cd LogAnomaly/template2Vector/src/preprocess/
    * python wordnet_process.py -data_dir ../../../middle/ -template_file kafka_log.template -syn_file kafka_log.syn -ant_file kafka_log.ant
* 参数样例：
    * data_dir：数据目录 ../../../middle/
    * template_file：模板文件 kafka_log.template
    * syn_file：同义词文件 kafka_log.syn
    * ant_file：反义词文件 kafka_log.ant

## 转换日志模板为词向量训练格式：
* 运行脚本的的命令：
    * cd LogAnomaly/middle/
    * python changeTemplateFormat.py -input kafka_log.template
* 参数样例：
    * input：模板文件 kafka_log.template

## 日志模板学习词向量(此操作需要在linux虚拟机执行)： 
* 运行脚本的的命令：
    * cd LogAnomaly/template2Vector/src/
    * make
    * ./lrcwe -train ../../middle/kafka_log.template_for_training -synonym ../../middle/kafka_log.syn -antonym ../../middle/kafka_log.ant -output ../../model/kafka_log.model -save-vocab ../../middle/kafka_log.vector_vocab -belta-rel 0.8 -alpha-rel 0.01 -belta-syn 0.4 -alpha-syn 0.2 -alpha-ant 0.3 -size 32 -min-count 1
* 参数样例：
    * train：训练文件 ../../middle/kafka_log.template_for_training
    * synonym：同义词文件 ../../middle/kafka_log.syn
    * antonym：反义词文件 ../../middle/kafka_log.ant
    * output：单词模型 ../../model/kafka_log.model
    * save-vocab：保存单词向量 ../../middle/kafka_log.vector_vocab
    * belta-rel 0.8 
    * alpha-rel 0.01 
    * belta-syn 0.4 
    * alpha-syn 0.2 
    * alpha-ant 0.3 
    * size 32 
    * min-count：最小词频训练阀值 1

## 获取模板向量：
* 运行脚本的的命令：
    * cd LogAnomaly/template2Vector/src/
    * python template2Vec.py -template_file ../../middle/kafka_log.template -word_model ../../model/kafka_log.model -template_vector_file ../../model/kafka_log.template_vector -dimension 32
* 参数样例：
    * template_file：模板文件 ../../middle/kafka_log.template
    * word_model：单词模型 ../../model/kafka_log.model
    * template_vector_file：模板向量文件 ../../model/kafka_log.template_vector
    * dimension：维度 32

## 模型训练和异常检测：
* 运行脚本的的命令：
    * cd LogAnomaly/LogAnomaly_main/
    * python -u train_vector_2LSTM_pytorch.py -data_file ../data/kafka_log.seq -label_file ../data/kafka_log.label  -seq_length 20 -n_candidates 20  -onehot 1 -template2Vec_file ../model/kafka_log.template_vector -count_matrix 0 -template_num 0 -template_file ../middle/kafka_log.template -epoch 100 -train_test_split 0.9
* 参数样例：
    * data_file：训练和验证数据集文件 ../data/kafka_log.seq
    * label_file：标签数据集文件 ../data/kafka_log.label
    * seq_length：序列长度 20
    * n_candidates：日志候选集数目 20
    * onehot：1:使用独热编码  0:不使用独热编码  1
    * template2Vec_file：模板向量文件 ../model/kafka_log.template_vector
    * count_matrix：1:统计count_matrix  0:不统计  0
    * template_file：模板文件 ../middle/kafka_log.template
    * template_num：若为0，则根据输入文件统计，否则，根据输入确定。默认0 0
    * epoch：模型训练轮数 100
    * train_test_split：训练数据和测试数据集划分比例 0.9

* 日志检测训练集和测试集比例划分如果是0.7或者0.8，由于一部分模板在训练中未学习到，导致准确率和召回率低，适当增加训练集数据比例，可以增加准确率和召回率。
检测结果:
查准率为93.67088607594937
召回率为100.0
F!为96.73202614379085

## 注意
1. 本项目在loganomaly开源项目基础上修改，从tensorflow项目修改为pytorch项目。
2. 数据集使用GAIA开源数据集中的日志检测数据集，只使用其中涉及到kafka的日志。已按照timestamp排序，相同timestamp，按照文件和行号排序。
3. 由于word2vec是已有方法，所用到的word2vec相关代码均直接使用Google提供的源码。
4. 由于本文重点在于异常检测，日志解析用到的是当前已有算法，所以用到的日志解析方法FT-tree的代码直接使用作者论文源码。

