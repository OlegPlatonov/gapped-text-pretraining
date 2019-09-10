import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import BertModel, BertPreTrainedModel, gelu, BertLayerNorm
from fairseq.models.roberta import RobertaModel


class BertGTHead(nn.Module):
    def __init__(self, hidden_size, window_size=15, two_layers=False):
        super(BertGTHead, self).__init__()
        self.dtype = torch.float32
        self.window_size = window_size
        self.two_layers = two_layers

        if window_size > 0:
            if not two_layers:
                self.gap_features_2_scores = nn.Linear(3 * hidden_size, 1)
                self.cls_features_2_scores = nn.Linear(3 * hidden_size, 1)
            else:
                self.gap_linear_1 = nn.Linear(3 * hidden_size, hidden_size)
                self.gap_linear_2 = nn.Linear(hidden_size, 1)
                self.cls_linear_1 = nn.Linear(3 * hidden_size, hidden_size)
                self.cls_linear_2 = nn.Linear(hidden_size, 1)
                self.GapLayerNorm = BertLayerNorm(hidden_size, eps=1.0e-12)
                self.ClsLayerNorm = BertLayerNorm(hidden_size, eps=1.0e-12)
        else:
            if not two_layers:
                self.gap_features_2_scores = nn.Linear(hidden_size, 1)
                self.cls_features_2_scores = nn.Linear(hidden_size, 1)
            else:
                self.gap_linear_1 = nn.Linear(hidden_size, hidden_size)
                self.gap_linear_2 = nn.Linear(hidden_size, 1)
                self.cls_linear_1 = nn.Linear(hidden_size, hidden_size)
                self.cls_linear_2 = nn.Linear(hidden_size, 1)
                self.GapLayerNorm = BertLayerNorm(hidden_size, eps=1.0e-12)
                self.ClsLayerNorm = BertLayerNorm(hidden_size, eps=1.0e-12)

    def forward(self, sequence_output, pooled_output, token_type_ids, word_mask, gap_ids):
        batch_size, seq_len, _ = sequence_output.shape
        device = sequence_output.device

        index_batch = torch.arange(batch_size).unsqueeze(1).to(device)
        index_seq = torch.arange(seq_len).unsqueeze(0).to(device)

        gaps = sequence_output[index_batch, gap_ids]

        if self.window_size > 0:
            # window pooling
            window_max_pool = []
            window_avg_pool = []
            for gap_id in torch.split(gap_ids, split_size_or_sections=1, dim=1):
                window_mask = (index_seq >= (gap_id - self.window_size)) * (index_seq <= (gap_id + self.window_size))
                window_mask = window_mask.type(torch.int64) * (token_type_ids == 0).type(torch.int64) * word_mask
                window = sequence_output * window_mask.unsqueeze(2).type(self.dtype)

                current_max_pool, _ = torch.max(window, dim=1, keepdim=True)
                num_tokens = torch.unsqueeze(torch.sum(window_mask, dim=1, keepdim=True), dim=2).type(self.dtype)
                current_avg_pool = torch.sum(window, dim=1, keepdim=True) / num_tokens

                window_max_pool.append(current_max_pool)
                window_avg_pool.append(current_avg_pool)

            window_max_pool = torch.cat(window_max_pool, dim=1)
            window_avg_pool = torch.cat(window_avg_pool, dim=1)

            # whole text pooling
            window_mask = (token_type_ids == 0).type(torch.int64) * word_mask
            window = sequence_output * window_mask.unsqueeze(2).type(self.dtype)

            text_max_pool, _ = torch.max(window, dim=1, keepdim=True)
            num_tokens = torch.unsqueeze(torch.sum(window_mask, dim=1, keepdim=True), dim=2).type(self.dtype)
            text_avg_pool = torch.sum(window, dim=1, keepdim=True) / num_tokens

            gap_features = torch.cat([gaps, window_max_pool, window_avg_pool], dim=-1)
            if pooled_output is not None:
                cls_features = torch.cat([pooled_output.unsqueeze(1), text_max_pool, text_avg_pool], dim=-1)
            else:
                cls_features = torch.cat([sequence_output[:, 0, :].unsqueeze(1), text_max_pool, text_avg_pool], dim=-1)

        else:
            gap_features = gaps
            if pooled_output is not None:
                cls_features = pooled_output.unsqueeze(1)
            else:
                cls_features = sequence_output[:, 0, :].unsqueeze(1)

        if not self.two_layers:
            gap_scores = self.gap_features_2_scores(gap_features).squeeze(-1)
            cls_scores = self.cls_features_2_scores(cls_features).squeeze(-1)
        else:
            gap_scores = self.gap_linear_2(self.GapLayerNorm(gelu(self.gap_linear_1(gap_features)))).squeeze(-1)
            cls_scores = self.cls_linear_2(self.ClsLayerNorm(gelu(self.cls_linear_1(cls_features)))).squeeze(-1)

        scores = torch.cat((cls_scores, gap_scores), dim=1)

        return scores


class BertForGappedText(BertPreTrainedModel):
    """BERT model with the head for Gapped Text task (with window pooling).
    This module comprises the BERT model followed by the masked language modeling head.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False
        'window_size': window size for window pooling. Default: 15
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `word_mask`: torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] where
            0 corresponds to special tokens (e.g. [CLS]) and 1 corresponda to all other tokens.
        'gap_ids': torch.LongTensor of shape [batch_size, num_gaps] with indices of [GAP] tokens in sequences.
        'target_gaps': torch.LongTensor of shape [batch_size] with values in [0, 1, ..., num_gaps]. These are target gaps
            for classification. 0 means that fragment is not from any of the gaps.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, window_size=15):
        super(BertForGappedText, self).__init__(config)
        self.bert = BertModel(config)
        self.output_layer = BertGTHead(config.hidden_size, window_size=window_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps=None, position_ids=None, head_mask=None, use_output_head=True):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)

        if use_output_head:
            sequence_output, pooled_output = outputs[:2]
        else:
            sequence_output = outputs[0]
            pooled_output = None

        gap_scores = self.output_layer(sequence_output=sequence_output,
                                       pooled_output=pooled_output,
                                       token_type_ids=token_type_ids,
                                       word_mask=word_mask,
                                       gap_ids=gap_ids)

        outputs = (gap_scores,) + outputs[2:]

        if target_gaps is not None:
            loss = F.cross_entropy(input=gap_scores, target=target_gaps)
            outputs = (loss,) + outputs

        return outputs


class RobertaForGappedText(nn.Module):
    def __init__(self, model_path, window_size=15, two_layers=False):
        super(RobertaForGappedText, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_path).model
        self.config = vars(self.roberta.args)
        self.output_layer = BertGTHead(self.roberta.args.encoder_embed_dim, window_size=window_size, two_layers=two_layers)

        if not two_layers:
            self.output_layer.gap_features_2_scores.weight.data.normal_(mean=0.0, std=0.02)
            self.output_layer.cls_features_2_scores.weight.data.normal_(mean=0.0, std=0.02)
            self.output_layer.gap_features_2_scores.bias.data.zero_()
            self.output_layer.cls_features_2_scores.bias.data.zero_()
        else:
            self.output_layer.gap_linear_1.weight.data.normal_(mean=0.0, std=0.02)
            self.output_layer.cls_linear_1.weight.data.normal_(mean=0.0, std=0.02)
            self.output_layer.gap_linear_1.bias.data.zero_()
            self.output_layer.cls_linear_1.bias.data.zero_()
            self.output_layer.gap_linear_2.weight.data.normal_(mean=0.0, std=0.02)
            self.output_layer.cls_linear_2.weight.data.normal_(mean=0.0, std=0.02)
            self.output_layer.gap_linear_2.bias.data.zero_()
            self.output_layer.cls_linear_2.bias.data.zero_()
            self.output_layer.GapLayerNorm.weight.data.fill_(1.0)
            self.output_layer.GapLayerNorm.bias.data.zero_()
            self.output_layer.ClsLayerNorm.weight.data.fill_(1.0)
            self.output_layer.ClsLayerNorm.bias.data.zero_()

    def forward(self, input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps=None, position_ids=None, head_mask=None, use_output_head=False):
        outputs = self.roberta(src_tokens=input_ids,
                               features_only=True)

        sequence_output = outputs[0]

        gap_scores = self.output_layer(sequence_output=sequence_output,
                                       pooled_output=None,
                                       token_type_ids=token_type_ids,
                                       word_mask=word_mask,
                                       gap_ids=gap_ids)

        outputs = (gap_scores,)

        if target_gaps is not None:
            loss = F.cross_entropy(input=gap_scores, target=target_gaps)
            outputs = (loss,) + outputs

        return outputs


class BertForGappedTextNoWindowPooling(BertPreTrainedModel):
    def __init__(self, config, window_size=0):
        super(BertForGappedTextNoWindowPooling, self).__init__(config)
        self.bert = BertModel(config)
        self.output_layer = BertGTHead(config.hidden_size, window_size=0)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps=None, position_ids=None, head_mask=None,
                use_output_head=True):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)

        if use_output_head:
            sequence_output, pooled_output = outputs[:2]
        else:
            sequence_output = outputs[0]
            pooled_output = None

        gap_scores = self.output_layer(sequence_output=sequence_output,
                                       pooled_output=pooled_output,
                                       token_type_ids=token_type_ids,
                                       word_mask=word_mask,
                                       gap_ids=gap_ids)

        outputs = (gap_scores,) + outputs[2:]

        if target_gaps is not None:
            loss = F.cross_entropy(input=gap_scores, target=target_gaps)
            outputs = (loss,) + outputs

        return outputs


class BertForGappedTextTwoLayers(BertPreTrainedModel):
    def __init__(self, config, window_size=15):
        super(BertForGappedTextTwoLayers, self).__init__(config)
        self.bert = BertModel(config)
        self.output_layer = BertGTHead(config.hidden_size, window_size=window_size, two_layers=True)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, word_mask, gap_ids, target_gaps=None, position_ids=None, head_mask=None,
                use_output_head=True):
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)

        if use_output_head:
            sequence_output, pooled_output = outputs[:2]
        else:
            sequence_output = outputs[0]
            pooled_output = None

        gap_scores = self.output_layer(sequence_output=sequence_output,
                                       pooled_output=pooled_output,
                                       token_type_ids=token_type_ids,
                                       word_mask=word_mask,
                                       gap_ids=gap_ids)

        outputs = (gap_scores,) + outputs[2:]

        if target_gaps is not None:
            loss = F.cross_entropy(input=gap_scores, target=target_gaps)
            outputs = (loss,) + outputs

        return outputs
