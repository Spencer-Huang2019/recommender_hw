import numpy as np
import random
import hashlib
import os
import zipfile

from download_utils import maybe_download


def data_process(train_file, out_f, n_samples):
    """
    :param train_file: archive.zip
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
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16) % (nr_bins-1)+1


def download_zip(data_path, azure_container_url, remote_resource_name):
    os.makedirs(data_path, exist_ok=True)
    remote_path = azure_container_url
    maybe_download(remote_path, remote_resource_name, data_path)
    if os.path.exists(remote_resource_name):
        zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), "r")
        zip_ref.extractall(data_path)
        zip_ref.close()
        os.remove(os.path.join(data_path, remote_resource_name))


def unzip_file(zip_src, dst_dir, clean_zip_file=False):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)


if __name__ == '__main__':
    data_path = "../data"
    filename = "archive.zip"
    download_url = "https://cloud.tsinghua.edu.cn/f/7b6e3e5e6ca545059503/?dl=1"
    download_zip(data_path, download_url, filename)

    zip_file = "../data/archive.zip"
    dst_path = "../data/train_data"
    unzip_file(zip_file, dst_path)

    origin_file = "../data/train_data/train_1m.txt"
    out_file1 = "../data/train_data/train_data"
    data_process(origin_file, out_file1, 200000)
    out_file2 = "../data/train_data/test_data"
    data_process(origin_file, out_file2, 10000)
    out_file3 = "../data/train_data/valid_data"
    data_process(origin_file, out_file3, 5000)
