from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pk
from itertools import chain
import random

import tokenization
from classifier_helper import DataProcessor, InputExample

# INPUT_PKL_FP = '/Users/king/Documents/Ein/Codes/dpf/data/projects/botlet_insurance/cv_0/dual_train_dev_samples.pkl'


class ForTestProcessor(DataProcessor):
    SEP = '||'

    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'classifier_train.txt')
        return self._get_examples(file_path)

    def _get_examples(self, file_path):
        with open(file_path, 'r') as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split(self.SEP)
            text_a = tokenization.convert_to_unicode(split_line[1])
            text_b = tokenization.convert_to_unicode(split_line[2])
            label = split_line[0]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'classifier_dev.txt')
        return self._get_examples(file_path)

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'classifier_dev.txt')
        return self._get_examples(file_path)

    def get_labels(self):
        return ['0', '1']


class QabotProcessor(DataProcessor):
    DATA_PKL_FILE = 'dual_train_dev_samples.pkl'
    TRAIN_SAMPLE_NUM = int(os.getenv('BERT_TRAIN_SAMPLE_NUM', '100'))
    DEV_SAMPLE_NUM = int(os.getenv('BERT_DEV_SAMPLE_NUM', '100'))

    def __init__(self):
        # input_pkl_fp = INPUT_PKL_FP
        # self.train, self.dev, self.data_info = pk.load(open(input_pkl_fp, 'rb'))
        self.train, self.dev, self.data_info = None, None, None
        print('TRAIN_SAMPLE_NUM: %d' % self.TRAIN_SAMPLE_NUM)
        print('DEV_SAMPLE_NUM: %d' % self.DEV_SAMPLE_NUM)

    def _get_ori_sample(self, data, limit=-1):
        pos_data = data['positives']
        neg_data = data['negatives']
        aid_list = list(set(neg_data.keys()) | set(pos_data.keys()))
        pos_samples = []
        neg_samples = []
        for aid in aid_list:
            part_pos_list = pos_data[aid]
            part_neg_dict = neg_data[aid]
            part_neg_list = sample_from_neg_dict(part_neg_dict, -1)
            pos_samples.extend(part_pos_list)
            neg_samples.extend(part_neg_list)
        if limit > 0:
            random.shuffle(pos_samples)
            random.shuffle(neg_samples)

            pos_num = max(limit // 2, 1)
            neg_num = limit - pos_num
            pos_samples = pos_samples[:pos_num]
            neg_samples = neg_samples[:neg_num]
        pos_samples.extend(neg_samples)
        return pos_samples

    def _get_examples(self, data, limit=-1):
        ori_samples = self._get_ori_sample(data, limit)
        questions = self.data_info['questions']

        examples = []
        for (i, sample) in enumerate(ori_samples):
            if i == 0:
                continue
            guid = "train-%d" % i
            text_a = tokenization.convert_to_unicode(questions[sample[0]])
            text_b = tokenization.convert_to_unicode(questions[sample[1]])
            label = tokenization.convert_to_unicode(str(sample[2]))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _load_data(self, data_dir):
        input_pkl_fp = os.path.join(data_dir, self.DATA_PKL_FILE)
        self.train, self.dev, self.data_info = pk.load(open(input_pkl_fp, 'rb'))

    def get_train_examples(self, data_dir):
        if self.train is None:
            self._load_data(data_dir)
        return self._get_examples(self.train, limit=self.TRAIN_SAMPLE_NUM)

    def get_dev_examples(self, data_dir):
        if self.dev is None:
            self._load_data(data_dir)
        return self._get_examples(self.dev, limit=self.DEV_SAMPLE_NUM)

    def get_test_examples(self, data_dir):
        """暂时就把dev看成test数据"""
        if self.dev is None:
            self._load_data(data_dir)
        return self._get_examples(self.dev, limit=self.DEV_SAMPLE_NUM)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


def sample_from_neg_dict(neg_dict, n):
    if not any(neg_dict.values()):
        # 所有的值都是空的话，就不可能抽取数据了
        return []

    neg_list = list(chain(*neg_dict.values()))
    random.shuffle(neg_list)
    if n > 0:
        return neg_list[:n]
    else:
        return neg_list
