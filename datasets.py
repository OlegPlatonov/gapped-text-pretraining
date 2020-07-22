import os
import gzip
import json
import yaml
import random
from itertools import islice
from string import punctuation
from abc import ABC, ABCMeta, abstractmethod

import numpy as np
import torch
import torch.utils.data as data


class DatasetRegistry(ABCMeta):
    registry = {}

    def __new__(mcs, name, bases, attrs):
        new_cls = ABCMeta.__new__(mcs, name, bases, attrs)
        mcs.registry[new_cls.task] = new_cls
        return new_cls

    @classmethod
    def get_dataset(mcs, task):
        return mcs.registry[task]


class BaseDataset(ABC, data.IterableDataset, metaclass=DatasetRegistry):
    task = None

    def __init__(self, data_file, data_size, tokenizer, local_rank, world_size=None, preprocessing_config=None):
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f'{data_file} does not exist or is a directory')

        self.data_file = data_file
        self.size = data_size

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.vocab[tokenizer.pad_token]

        if preprocessing_config is not None:
            with open(preprocessing_config, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = {}

        if local_rank == -1:
            self.start = 0
            self.step = 1
        else:
            self.start = local_rank
            self.step = world_size

    def __len__(self):
        return self.size

    def __iter__(self):
        if self.data_file.endswith('.gz'):
            file_iter = gzip.open(self.data_file, 'rt')
        else:
            file_iter = open(self.data_file, 'r')

        islice_iter = islice(file_iter, self.start, None, self.step)
        loaded_iter = map(json.loads, islice_iter)
        processed_iter = map(self.process_line, loaded_iter)
        return processed_iter

    @staticmethod
    def tokenize_first_segment(segment):
        return ['<s>'] + segment.split() + ['</s>']

    @staticmethod
    def tokenize_second_segment(segment):
        return ['</s>'] + segment.split() + ['</s>']

    @staticmethod
    def pad_2d(array_2d, pad_value=0):
        row_lengths = [len(row) for row in array_2d]
        max_len = max(row_lengths)
        for i in range(len(array_2d)):
            array_2d[i] += [pad_value for _ in range(max_len - row_lengths[i])]

        return array_2d

    @classmethod
    @abstractmethod
    def process_line(cls, line):
        pass

    @classmethod
    @abstractmethod
    def collate_fn(cls, batch):
        pass


class DatasetForGT(BaseDataset):
    task = 'GT'

    def __init__(self, data_file, data_size, tokenizer, local_rank, world_size=None, preprocessing_config=None):
        super(DatasetForGT, self).__init__(data_file, data_size, tokenizer, local_rank, world_size,
                                           preprocessing_config)
        self.gap_token_id = self.tokenizer.convert_tokens_to_ids(['<gap>'])[0]

    @classmethod
    def process_line(cls, line):
        text = line['text']
        fragment = line['fragment']
        target_gap = int(line['target_gap'])

        fragment_sequence = cls.tokenize_first_segment(fragment)
        fragment_sequence = cls.tokenizer.convert_tokens_to_ids(fragment_sequence)
        text_sequence = cls.tokenize_second_segment(text)
        text_sequence = cls.tokenizer.convert_tokens_to_ids(text_sequence)
        input_ids = fragment_sequence + text_sequence
        attention_mask = [1 for _ in input_ids]
        gap_ids = [i for i in range(len(input_ids)) if input_ids[i] == cls.gap_token_id]

        return input_ids, attention_mask, gap_ids, target_gap

    @classmethod
    def collate_fn(cls, batch):
        input_ids = []
        attention_mask = []
        gap_ids = []
        target_gaps = []

        for cur_input_ids, cur_attention_mask, cur_gap_ids, cur_target_gap in batch:
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            gap_ids.append(cur_gap_ids)
            target_gaps.append(cur_target_gap)

        input_ids = torch.tensor(cls.pad_2d(input_ids, pad_value=cls.pad_token_id))
        attention_mask = torch.tensor(cls.pad_2d(attention_mask))
        gap_ids = torch.tensor(gap_ids)
        target_gaps = torch.tensor(target_gaps)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'gap_ids': gap_ids,
                'target_gaps': target_gaps}


class DatasetForQA(BaseDataset):
    task = 'QA'

    @classmethod
    def process_line(cls, line):
        text = line['text']
        question = line['question']
        answer_start = int(line['answer_start'])
        answer_end = int(line['answer_end'])

        question_sequence = cls.tokenize_first_segment(question)
        question_sequence = cls.tokenizer.convert_tokens_to_ids(question_sequence)
        text_sequence = cls.tokenize_second_segment(text)
        text_sequence = cls.tokenizer.convert_tokens_to_ids(text_sequence)
        input_ids = question_sequence + text_sequence
        attention_mask = [1 for _ in input_ids]
        answer_start = 0 if answer_start == -1 else answer_start + len(question_sequence) + 1
        answer_end = 0 if answer_end == -1 else answer_end + len(question_sequence) + 1

        return input_ids, attention_mask, answer_start, answer_end

    def collate_fn(cls, batch):
        input_ids = []
        attention_mask = []
        answer_start = []
        answer_end = []

        for cur_input_ids, cur_attention_mask, cur_answer_start, cur_answer_end in batch:
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            answer_start.append(cur_answer_start)
            answer_end.append(cur_answer_end)

        input_ids = torch.tensor(cls.pad_2d(input_ids, pad_value=cls.pad_token_id))
        attention_mask = torch.tensor(cls.pad_2d(attention_mask))
        answer_start = torch.tensor(answer_start)
        answer_end = torch.tensor(answer_end)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'answer_start': answer_start,
                'answer_end': answer_end}


def find_words_to_mask(sequence, mask_proportion, topic_words=(), topic_mask_proportion=1):
    word_ids = []
    for i, token in enumerate(sequence):
        if (token.startswith('[') and token.endswith(']')) or token in punctuation:
            continue
        elif token.startswith('##') and word_ids:
            word_ids[-1][1] += 1
        else:
            word_ids.append([i, 1])

    if not topic_words:
        num_to_mask = int(len(word_ids) * mask_proportion)
        word_ids_to_mask = random.sample(word_ids, num_to_mask)
        return sorted(word_ids_to_mask)

    topic_word_ids = []
    other_word_ids = []
    for word_id in word_ids:
        word_start = sequence[word_id[0]]
        for topic_word in topic_words:
            if topic_word in word_start:
                topic_word_ids.append(word_id)
                break
        else:
            other_word_ids.append(word_id)

    words_to_mask = []
    topic_words_mask = np.random.binomial(n=1, p=topic_mask_proportion, size=len(topic_word_ids))
    for i, topic_word_id in enumerate(topic_word_ids):
        if topic_words_mask[i]:
            words_to_mask.append(topic_word_id)

    num_to_mask = int(mask_proportion * len(word_ids)) - len(words_to_mask)
    if num_to_mask > 0:
        words_to_mask.extend(random.sample(other_word_ids, num_to_mask))

    return sorted(words_to_mask)


def mask_words(sequence, words_to_mask, mask_token_id, shift=0):
    mask_ids = []
    mask_targets = []
    for start_idx, length in words_to_mask:
        for i in range(length):
            current_idx = start_idx + i + shift
            mask_ids.append(current_idx)
            mask_targets.append(sequence[current_idx])
            sequence[current_idx] = mask_token_id

    return mask_ids, mask_targets


class DatasetForMLM(BaseDataset):
    task = 'MLM'

    def __init__(self, data_file, data_size, tokenizer, local_rank, world_size=None, preprocessing_config=None):
        super(DatasetForMLM, self).__init__(data_file, data_size, tokenizer, local_rank, world_size,
                                            preprocessing_config)
        self.mask_token_id = tokenizer.vocab[tokenizer.mask_token]
        self.mask_proportion = self.config['mask_proportion']
        self.topic_words = self.config['topic_words'] if 'topic_words' in self.config else ()
        self.topic_mask_proportion = (self.config['topic_mask_proportion'] if 'topic_mask_proportion' in self.config
                                      else 1)

    def process_line(self, line):
        text_sequence = line['text'].split()
        words_to_mask = find_words_to_mask(text_sequence, self.mask_proportion,
                                           self.topic_words, self.topic_mask_proportion)

        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text_sequence + ['[SEP]'])
        attention_mask = [1 for _ in input_ids]
        mask_ids, mask_targets = mask_words(input_ids, words_to_mask, self.mask_token_id, shift=1)

        return input_ids, attention_mask, mask_ids, mask_targets

    def collate_fn(self, batch):
        input_ids = []
        attention_mask = []
        mask_ids = []
        mask_targets = []

        for cur_input_ids, cur_attention_mask, cur_mask_ids, cur_mask_targets in batch:
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            mask_ids.append(cur_mask_ids)
            mask_targets.append(cur_mask_targets)

        input_ids = torch.tensor(self.pad_2d(input_ids, pad_value=self.pad_token_id))
        attention_mask = torch.tensor(self.pad_2d(attention_mask))
        mask_ids = torch.tensor([[row, col] for row, line in enumerate(mask_ids) for col in line])
        mask_targets = torch.tensor([token_idx for line in mask_targets for token_idx in line])

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'mask_ids': mask_ids,
                'mask_targets': mask_targets}
