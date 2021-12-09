import numpy as np
import random
import hashlib

def sampling(train_file, out_f, n_samples):
    """
    对用户和电影进行随机抽样
    :param train_file: train_1m.txt
    :param n_samples: 抽样数量
    :return: 返回抽样后的数据集
    """
    with open(train_file, 'r') as f1:
        data_list = f1.read().splitlines()
        records = [ele for ele in data_list]
        # 随机抽样
        sample_list = random.sample(records, n_samples)
        sample_str_matrix = []
        for record in sample_list:
            record_list = record.split('\t')
            r_list = []
            for y in range(0, 40):
                if y == 0:
                    r_list.append(record_list[0])
                else:
                    processed_rec = de_hash(record_list[y])
                    r_list.append(processed_rec)
            sample_str_matrix.append(r_list)
        # print(sample_str_matrix)

        data_set = []
        j = 0
        for rec in sample_str_matrix:
            data_set.append([])
            for l in range(0, 40):
                if l == 0:
                    data_set[j].append(rec[0])
                else:
                    k = str(l)
                    # print(e)
                    value = rec[l]
                    data_set[j].append(k + ":" + str(value) + ":" + '1')
            j += 1
        print(data_set)

    with open(out_f, 'w') as f:
        for x1 in data_set:
            # 每一行的元素用空格隔开，且每一行是一个字符串
            f.write(' '.join(str(x2) for x2 in x1))
            f.write('\n')


def de_hash(str, nr_bins=6000):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1


if __name__ == '__main__':
    origin_file = "../data/train_data/train_1m.txt"
    out_file = "../data/train_data/train_data"
    sampling(origin_file, out_file, 200000)