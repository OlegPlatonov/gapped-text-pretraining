import random
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

fragment_transforms = {
    'bert-base-uncased': lambda fragment: ['[CLS]'] + fragment.split() + ['[SEP]'],
    'roberta': lambda fragment: ['<s>'] + fragment.split() + ['</s>', '</s>']
}

text_transforms = {
    'bert-base-uncased': lambda text: text.split() + ['[SEP]'],
    'roberta': lambda text: text.split() + ['</s>']
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
        return len(self.data['segment_1']) * 2

    def __getitem__(self, idx):
        text_idx = idx // 2
        swap = (idx % 2 == 1)

        segment_1 = self.data['segment_1'][text_idx]
        segment_2 = self.data['segment_2'][text_idx]
        if swap:
            segment_1, segment_2 = segment_2, segment_1

        target_order = 0 if swap else 1

        return segment_1, segment_2, target_order


def pad_2d(array_2d, pad_value=0):
    row_lengths = [len(row) for row in array_2d]
    max_len = max(row_lengths)
    for i in range(len(array_2d)):
        array_2d[i] += [pad_value for _ in range(max_len - row_lengths[i])]

    return array_2d


def find_words_to_mask(sequence, mask_proportion):
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

    num_to_mask = int(len(words) * mask_proportion)
    words_to_mask = sorted(random.sample(words, num_to_mask))

    return words_to_mask


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


def GT_collate_fn(batch, mask_proportion):
    model_type = 'roberta'

    input_ids = []
    token_type_ids = []
    attention_mask = []
    word_mask = []
    sop_targets = []
    mask_ids = []
    mask_targets = []

    for segment_1, segment_2, target_order in batch:
        sequence_1 = fragment_transforms[model_type](segment_1)
        words_to_mask_1 = find_words_to_mask(sequence_1, mask_proportion)
        sequence_1 = tokenizers[model_type].convert_tokens_to_ids(sequence_1)

        sequence_2 = text_transforms[model_type](segment_2)
        words_to_mask_2 = find_words_to_mask(sequence_2, mask_proportion)
        sequence_2 = tokenizers[model_type].convert_tokens_to_ids(sequence_2)

        full_sequence = sequence_1 + sequence_2

        input_ids.append(full_sequence)
        token_type_ids.append([0 for _ in range(len(sequence_1))] + [1 for _ in range(len(sequence_2))])
        attention_mask.append([1 for _ in full_sequence])
        word_mask.append([1 if idx not in special_token_ids[model_type] else 0 for idx in full_sequence])
        sop_targets.append(target_order)

        current_mask_ids_1, current_mask_targets_1 = mask_words(full_sequence, words_to_mask_1, mask_token_ids[model_type], shift=0)
        current_mask_ids_2, current_mask_targets_2 = mask_words(full_sequence, words_to_mask_2, mask_token_ids[model_type], shift=len(sequence_1))
        mask_ids.append(current_mask_ids_1 + current_mask_ids_2)
        mask_targets.append(current_mask_targets_1 + current_mask_targets_2)

    input_ids = torch.tensor(pad_2d(input_ids, pad_value=pad_token_ids[model_type]))
    token_type_ids = torch.tensor(pad_2d(token_type_ids, pad_value=1))
    attention_mask = torch.tensor(pad_2d(attention_mask))
    word_mask = torch.tensor(pad_2d(word_mask))
    sop_targets = torch.tensor(sop_targets)
    mask_ids = torch.tensor([[row, col] for row, line in enumerate(mask_ids) for col in line])
    mask_targets = torch.tensor([token_idx for line in mask_targets for token_idx in line])

    return input_ids, token_type_ids, attention_mask, word_mask, sop_targets, mask_ids, mask_targets

