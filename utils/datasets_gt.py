import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from transformers import BertTokenizer, RobertaTokenizer

bert_tokenizer = BertTokenizer('./models/vocabs/bert-base-uncased-vocab.txt',
                               additional_special_tokens=['[GAP]'],
                               do_basic_tokenize=False)

roberta_tokenizer = RobertaTokenizer('./models/vocabs/roberta-large-vocab.json',
                                     './models/vocabs/roberta-large-merges.txt',
                                     additional_special_tokens=['<gap>'],
                                     do_basic_tokenize=False)

tokenizers = {
    'bert-base-uncased': bert_tokenizer,
    'roberta': roberta_tokenizer
}

special_tokens = {
    'bert-base-uncased': ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[GAP]'],
    'roberta': ['<s>', '<pad>', '</s>', '<unk>', '<gap>']
}

special_token_ids = {
    'bert-base-uncased': tokenizers['bert-base-uncased'].convert_tokens_to_ids(special_tokens['bert-base-uncased']),
    'roberta': tokenizers['roberta'].convert_tokens_to_ids(special_tokens['roberta'])
}

gap_tokens = {
    'bert-base-uncased': '[GAP]',
    'roberta': '<gap>'
}

pad_tokens = {
    'bert-base-uncased': '[PAD]',
    'roberta': '<pad>'
}

gap_token_ids = {
    'bert-base-uncased': tokenizers['bert-base-uncased'].convert_tokens_to_ids(['[GAP]'])[0],
    'roberta': tokenizers['roberta'].convert_tokens_to_ids(['<gap>'])[0]
}

pad_token_ids = {
    'bert-base-uncased': tokenizers['bert-base-uncased'].convert_tokens_to_ids(['[PAD]'])[0],
    'roberta': tokenizers['roberta'].convert_tokens_to_ids(['<pad>'])[0]
}

mask_token_ids = {
    'bert-base-uncased': tokenizers['bert-base-uncased'].convert_tokens_to_ids(['[MASK]'])[0],
    'roberta': tokenizers['roberta'].convert_tokens_to_ids(['<mask>'])[0]
}

text_transforms = {
    'bert-base-uncased': lambda fragment: ['[CLS]'] + fragment.split() + ['[SEP]'],
    'roberta': lambda fragment: ['<s>'] + fragment.split() + ['</s>']
}


class GT_Dataset(data.Dataset):
    def __init__(self, data_path):
        super(GT_Dataset, self).__init__()
        self.data = dict()
        self.data['segment_1'] = list(pd.read_csv(data_path, usecols=['segment_1'], squeeze=True,
                                                  dtype='str', engine='c'))
        self.data['segment_2'] = list(pd.read_csv(data_path, usecols=['segment_2'], squeeze=True,
                                                  dtype='str', engine='c'))

    def __len__(self):
        return len(self.data['segment_1'])

    def __getitem__(self, idx):
        segment_1 = self.data['segment_1'][idx]
        segment_2 = self.data['segment_2'][idx]
        return ' '.join([segment_1, segment_2])


def pad_2d(array_2d, pad_value=0):
    row_lengths = [len(row) for row in array_2d]
    max_len = max(row_lengths)
    for i in range(len(array_2d)):
        array_2d[i] += [pad_value for _ in range(max_len - row_lengths[i])]

    return array_2d


def get_span_lengths(num_words, mask_proportion):
    lengths = []
    total_length = 0
    while total_length < num_words * mask_proportion:
        new_length = min([np.random.geometric(p=0.25, size=1)[0], 8])
        lengths.append(new_length)
        total_length += new_length

    return lengths, total_length



def find_words_to_mask(sequence, mask_proportion, logger):
    words = []
    prev_token_is_special = False
    for i, token in enumerate(sequence):
        if token.startswith('<') and token.endswith('>'):
            prev_token_is_special = True
        elif token.startswith('Ġ') or token in '.,:;"?!' or not words:
            words.append([i, 1])
            prev_token_is_special = False
        else:
            if not prev_token_is_special:
                words[-1][1] += 1

    span_lengths, total_length = get_span_lengths(len(words), mask_proportion)
    if total_length > len(words) // 2:
        span_lengths = []
        total_length = 0
        logger.warning('Spans are to long! Will not use span masking for this example.')

    num_options = len(words) - (total_length - len(span_lengths)) - len(span_lengths) * 2 - 2
    span_start_indices = sorted(random.sample(list(range(num_options)), len(span_lengths)))
    for i in reversed(range(1, len(span_start_indices))):
        span_start_indices[i] -= span_start_indices[i - 1]

    words_to_mask = []
    span_features = []
    current_id = 2
    for i, start_idx in enumerate(span_start_indices):
        current_id += start_idx

        start_word_id = current_id
        prev_token_id = words[current_id][0] - 1

        for _ in range(span_lengths[i]):
            words_to_mask.append(words[current_id])
            current_id += 1

        next_token_id = words[current_id][0]

        token_position = 0
        for word_id in range(start_word_id, current_id):
            for _ in range(words[word_id][1]):
                span_features.append([prev_token_id, next_token_id, token_position])
                token_position += 1

        current_id += 1

    return words_to_mask, span_features


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


def GT_collate_fn(batch, mask_proportion, logger):
    model_type = 'roberta'

    input_ids = []
    token_type_ids = []
    attention_mask = []
    word_mask = []
    mask_ids = []
    span_features = []
    mask_targets = []

    for segment in batch:
        sequence = text_transforms[model_type](segment)
        words_to_mask, current_span_features = find_words_to_mask(sequence, mask_proportion, logger)
        sequence = tokenizers[model_type].convert_tokens_to_ids(sequence)

        input_ids.append(sequence)
        token_type_ids.append([0 for _ in sequence])
        attention_mask.append([1 for _ in sequence])
        word_mask.append([1 if idx not in special_token_ids[model_type] else 0 for idx in sequence])

        current_mask_ids, current_mask_targets = mask_words(sequence, words_to_mask, mask_token_ids[model_type], shift=0)
        mask_ids.append(current_mask_ids)
        span_features.append(current_span_features)
        mask_targets.append(current_mask_targets)

    input_ids = torch.tensor(pad_2d(input_ids, pad_value=pad_token_ids[model_type]))
    token_type_ids = torch.tensor(pad_2d(token_type_ids, pad_value=1))
    attention_mask = torch.tensor(pad_2d(attention_mask))
    word_mask = torch.tensor(pad_2d(word_mask))
    mask_ids = torch.tensor([[row, col] for row, line in enumerate(mask_ids) for col in line])
    span_features = torch.tensor([[row, prev_id, next_id, position] for row, line in enumerate(span_features) for prev_id, next_id, position in line])
    mask_targets = torch.tensor([token_idx for line in mask_targets for token_idx in line])

    return input_ids, token_type_ids, attention_mask, word_mask, mask_ids, span_features, mask_targets

