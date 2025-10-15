#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import argparse
from keras.utils.np_utils import to_categorical
from early_stopping import EarlyStopping
from template2vec import Template2Vec
import time
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dir(path):
    ''' 创建目录
    Args:
        path: 目录
    Return:
    '''
    import os
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def data_prepare(datafilename, labelfilename):
    # 训练集数据保证都是正常数据集 测试集数据需要一部分normal 一部分anormal
    all_data = []
    with open(datafilename) as line_IN:
        with open(labelfilename) as label_IN:
            for line, label_line in zip(line_IN, label_IN):
                all_data.append((line.strip() + " " + label_line.strip()).split())

    all_data = np.array(all_data)

    # 训练数据集 如果相邻两条日志时间间隔在一分钟之内，那么算为一组数据
    timestamps = all_data[:, 0]
    all_data_with_group = []
    start = 0
    end = 1
    while (end < len(all_data)):
        if int(timestamps[end]) - int(timestamps[end - 1]) > 60000:
            # 过滤组里面只有5条数据以内的组
            if end - start <= 5:
                end += 1
                start += 1
                continue
            all_data_with_group.append(all_data[start:end])
            start = end
            end += 1
        else:
            end += 1
    if end != start:
        all_data_with_group.append(all_data[start:])


    return all_data,all_data_with_group


def get_alpha(all_data, template_to_int, n_templates):
    """
    计算损失函数参数alpha权重列表
    Args:
        all_data:

    Returns:

    """
    alpha = []
    template_count = {}
    for i in range(len(all_data)):
        template = all_data[i][1]
        index = template_to_int[template]
        if index not in template_count.keys():
            template_count[index] = 0
        template_count[index] += 1

    sorted_tuples = sorted(template_count.items(), key=lambda kv: kv[0])
    # print(sorted_tuples)
    for kv in sorted_tuples:
        alpha.append(1-float(float(kv[1])/float(n_templates)))
    return alpha



def data_group(data_group, seq_length, temp2Vec, template_to_int, count_matrix_flag, n_templates, onehot):
    # 把数据按照模型输入格式划分组
    dataY = []
    dataLabel = []
    vectorX = []
    vectorY = []
    for data in data_group:
        raw_text = data[:, 1]
        label_list = data[:, 2]
        if len(raw_text) < seq_length + 1:
            # 如果组数据长度小于seq_length 那么前面扩充raw_text[0]
            raw_text = np.insert(raw_text, 0, [raw_text[0]] * (seq_length + 1 - len(raw_text)))
            # 如果组数据长度小于seq_length 那么前面扩充label_list[0]
            label_list = np.insert(label_list, 0, [label_list[0]] * (seq_length + 1 - len(label_list)))
        n_chars = len(raw_text)
        for i in range(0, n_chars - seq_length, 1):
            label = label_list[i + seq_length]
            dataLabel.append(label)
            seq_in = raw_text[i:i + seq_length]
            seq_out = raw_text[i + seq_length]
            dataY.append(template_to_int[seq_out])
            temp_list = []
            for seq in seq_in:
                if count_matrix_flag == 0:
                    # 不拼接，直接用template vector
                    temp_list.append(list(temp2Vec.model[seq]))
                else:
                    # 拼接template vector和count vector
                    cur_count_vector = [0 for i in range(n_templates)]
                    for t in seq_in:
                        cur_index = template_to_int[t]
                        cur_count_vector[cur_index] += 1

                    l = list(temp2Vec.model[seq])
                    l.extend(cur_count_vector)
                    temp_list.append(l)

            vectorX.append(np.array(temp_list))
            vectorY.append(temp2Vec.model[seq_out])

    n_patterns = len(vectorX)
    print("# of patterns:", n_patterns)

    if count_matrix_flag == 0:
        X = np.reshape(vectorX, (-1, seq_length, temp2Vec.dimension))
    else:
        X = np.reshape(vectorX, (-1, seq_length, temp2Vec.dimension + n_templates))

    if onehot == 1:
        # y = to_categorical(dataY, num_classes=n_templates)
        y = np.array(dataY)
        class_num = n_templates
    else:
        y = np.reshape(vectorY, (-1, temp2Vec.dimension))
        class_num = temp2Vec.dimension
    dataLabel = np.array(dataLabel)

    return X, y, dataLabel, class_num


# 定义模型
class templateVectorModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, class_num):
        super(templateVectorModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # droupout层 0.2
        self.droupout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, class_num)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.droupout(out)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


# 定义多分类损失函数
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1)  # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


def batch_data(data_x, data_y, data_label, batchsize=64):
    batch_data_x = []
    batch_data_y = []
    batch_data_label = []
    # 用于训练数据分组
    for i in range(int(len(data_x) // batchsize)):
        batch_data_x.append(data_x[i * batchsize:(i + 1) * batchsize])
        batch_data_y.append(data_y[i * batchsize:(i + 1) * batchsize])
        batch_data_label.append(data_label[i * batchsize:(i + 1) * batchsize])
    # if int(len(data_x) // batchsize) != 0:
    #     batch_data_x.append(data_x[0 - len(data_x) % batchsize:])
    #     batch_data_y.append(data_y[0 - len(data_x) % batchsize:])
    #     batch_data_label.append(data_label[0 - len(data_x) % batchsize:])
    return batch_data_x, batch_data_y, batch_data_label


def train_model(para):
    ''' 训练模型
    Args:
        para: 参数
    Return:
    '''
    epoch = para['epoch']
    datafilename = para['data_file']
    labelfilename = para['label_file']
    # n_candidates = para['n_candidates']
    template2Vec_file = para['template2Vec_file']
    tempalte_file = para['template_file']
    # model_dir = para['model_dir']
    train_test_split = para['train_test_split']
    # create_dir(model_dir)
    temp2Vec = Template2Vec(template2Vec_file, tempalte_file)
    all_data, all_data_with_group = data_prepare(datafilename, labelfilename)


    # 创建模板id和序列号关系文件
    seq_length = para['seq_length']
    template_num = para['template_num']

    count_matrix_flag = para['count_matrix']
    template_index_map_path = para['template_index_map_path']  # 保存模板号与向量的映射关系
    onehot = para['onehot']

    raw_text = all_data[:, 1]

    if template_num == 0:
        # 如果template_num为0，则根据模板序列文件来生成映射
        chars = sorted(list(set(raw_text)))
        template_to_int = dict((c, i) for i, c in enumerate(chars))
        print('template_to_int', template_to_int)
        f = open(template_index_map_path, 'w')
        for k in template_to_int:
            f.writelines(str(k) + ' ' + str(template_to_int[k]) + '\n')
        f.close()
    else:
        # 如果template_num不为0，则根据其构造映射,int从0开始，char从1开始
        template_to_int = dict((str(i + 1), i) for i in range(template_num))
        print('template_to_int', template_to_int)

    n_chars = len(raw_text)
    n_templates = len(template_to_int)
    # 取相似度前百分之10 的模板作为评价标准
    n_candidates = int(n_templates * 0.1)
    print("length of log sequence: ", n_chars)
    print("# of templates: ", n_templates)
    print("# of candidates: ", n_candidates)

    alpha = get_alpha(all_data, template_to_int,n_templates)

    all_X, all_y, all_dataLabel, class_num = data_group(all_data_with_group, seq_length, temp2Vec,
                                                              template_to_int,
                                                              count_matrix_flag, n_templates, onehot)

    train_X = all_X[:int(len(all_X)*train_test_split)]
    train_y = all_y[:int(len(all_X)*train_test_split)]
    train_dataLabel = all_dataLabel[:int(len(all_X)*train_test_split)]
    test_X = all_X[int(len(all_X)*train_test_split):]
    test_y = all_y[int(len(all_X)*train_test_split):]
    test_dataLabel = all_dataLabel[int(len(all_X)*train_test_split):]

    input_size = train_X.shape[2]
    hiddent_size = 128

    # 定义模型
    model = templateVectorModel(input_size=input_size, hidden_size=hiddent_size, num_layers=2, class_num=class_num)
    model.to(device)
    # 定义损失函数
    if onehot == 1:
        # 如果onehot为1 那么使用交叉熵损失函数
        # fn_loss = nn.CrossEntropyLoss()
        # focalloss
        fn_loss = MultiClassFocalLossWithAlpha(alpha)
    else:
        fn_loss = nn.MSELoss()
        # 实例化余弦相似度损失并计算损失
        # fn_loss = MultiClassFocalLossWithAlpha(alpha)

    fn_loss.to(device)
    # 定义优化器
    lr = 0.001
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    batch_train_data_x, batch_train_data_y, batch_train_data_label = batch_data(train_X, train_y, train_dataLabel,
                                                                                batchsize=64)
    batch_test_data_x, batch_test_data_y, batch_test_data_label = batch_data(test_X, test_y, test_dataLabel,
                                                                             batchsize=64)
    template_to_int = {}
    int_to_template = {}
    if template_num == 0:
        # 如果template_num为0，则根据模板序列文件来生成映射
        with open(template_index_map_path) as IN:
            for line in IN:
                l = line.strip().split()
                c = l[0]
                i = int(l[1])
                template_to_int[c] = i
                int_to_template[i] = c
    else:
        # 如果template_num不为0，则根据其构造映射,int从0开始，char从1开始
        template_to_int = dict((str(i + 1), i) for i in range(template_num))
        int_to_template = dict((i, str(i + 1)) for i in range(template_num))
    total_train_step = 0
    start_time = time.time()
    # 初始化早停止对象
    save_path = "../model"
    early_stopping = EarlyStopping(save_path)
    for step in range(epoch):
        # 开始训练
        model.train()
        for i in range(len(batch_train_data_x)):
            data_x = batch_train_data_x[i]
            data_x = torch.tensor(data_x.tolist())
            data_x = data_x.to(device)
            data_y = batch_train_data_y[i]
            data_y = torch.tensor(data_y.tolist())
            data_y = data_y.to(device)
            y_predict = model(data_x)
            loss = fn_loss(y_predict, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 10 == 0:
                end_time = time.time()
                print(end_time - start_time)
                print("total train step is {}，loss is {}".format(total_train_step, loss.item()))
        # 开始测试
        TP = 0
        FP = 0
        FN = 0
        with torch.no_grad():
            model.eval()
            for i in range(len(batch_test_data_x)):
                data_x = batch_test_data_x[i]
                data_x = torch.tensor(data_x.tolist())
                data_x = data_x.to(device)
                data_y = batch_test_data_y[i]
                y_predict = model(data_x)
                data_label = batch_test_data_label[i]
                for j in range(len(data_y)):
                    data_y_single = data_y[j]
                    # 获取最相似的topn
                    if onehot == 1:
                        # aim_y_char = int_to_template[torch.tensor(data_y_single.tolist()).argsort()[-1:].tolist()[0]]
                        aim_y_char = int_to_template[data_y_single]
                        top_n_index = y_predict[j].argsort()[-n_candidates:].tolist()
                        top_n_tag = [int_to_template[index] for index in top_n_index]
                    else:
                        aim_y_char = temp2Vec.vector_to_most_similar(data_y_single, topn=1)[0][0]
                        top_n_tuple = temp2Vec.vector_to_most_similar(np.array(y_predict[j].tolist()),
                                                                      topn=n_candidates)
                        top_n_tag = [top_single_tuple[0] for top_single_tuple in top_n_tuple]
                    label = int(data_label[j])
                    if label == 0:
                        if aim_y_char not in top_n_tag:
                            FP += 1
                    else:
                        if aim_y_char not in top_n_tag:
                            TP += 1
                        else:
                            FN += 1
        elapsed_time = time.time() - start_time
        print('第{}轮模型训练与评估耗时: {}'.format(step + 1, elapsed_time))
        # Compute precision, recall and F1-measure

        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print("查准率为{}".format(P))
        print("召回率为{}".format(R))
        print("F!为{}".format(F1))


        # 早停止
        early_stopping(P, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    end_time = time.time()
    print("all trining is finished!,spend time is {}".format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_file', help='data_file.', type=str, default='../data/kafka_log.seq')
    parser.add_argument('-label_file', help='label_file.', type=str, default='../data/kafka_log.label')
    parser.add_argument('-seq_length', help='seq_length.', type=int, default=20)
    parser.add_argument('-n_candidates', help='n_candidates.', type=int, default=20)
    # parser.add_argument('-model_dir', help='网络参数的输出文件夹', type=str, default='../weights/vector_deeplog/')
    parser.add_argument('-template_num', help='若为0，则根据输入文件统计，否则，根据输入确定。默认0', type=int, default=0)
    parser.add_argument('-template2Vec_file', help='template2Vec_file', type=str,
                        default='../model/kafka_log.template_vector')
    parser.add_argument('-count_matrix', help='默认为0。1表示统计count_matrix，0不统计', type=int, default=0)
    parser.add_argument('-onehot', help='默认为1。1表示统计使用onehot，0表示使用template2vec', type=int, default=1)
    parser.add_argument('-template_file', help='template_file', type=str, default='../middle/kafka_log.template')
    parser.add_argument('-epoch', help='epoch', type=int, default=100)
    parser.add_argument('-train_test_split', help='train_test_split', type=float, default=0.9)

    args = parser.parse_args()

    para_train = {
        'data_file': args.data_file,
        'label_file': args.label_file,
        'seq_length': args.seq_length,
        'n_candidates': args.n_candidates,
        'model_dir': args.model_dir,
        'template_index_map_path': args.data_file + '_map',
        'template_num': args.template_num,
        'template2Vec_file': args.template2Vec_file,
        'template_file': args.template_file,
        'count_matrix': args.count_matrix,
        'onehot': args.onehot,
        'epoch': args.epoch,
        'train_test_split': args.train_test_split
    }

    train_model(para_train)
