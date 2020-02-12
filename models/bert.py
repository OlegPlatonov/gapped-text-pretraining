import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_roberta import RobertaModel, RobertaLMHead
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_bert import BertPreTrainedModel, BertLayerNorm, gelu_new


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base":
        "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class GT_Head(nn.Module):
    def __init__(self, hidden_size):
        super(GT_Head, self).__init__()
        self.features_2_scores = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output, gap_ids):
        batch_size, seq_len, _ = sequence_output.shape
        device = sequence_output.device

        index_batch = torch.arange(batch_size).unsqueeze(1).to(device)
        gap_ids = torch.cat([torch.zeros((batch_size, 1)).to(device).long(), gap_ids], dim=1)
        gaps = sequence_output[index_batch, gap_ids]

        gap_scores = self.features_2_scores(gaps).squeeze(-1)

        return gap_scores


class SimpleLayer(nn.Module):
    def __init__(self, config, num_inputs):
        super(SimpleLayer, self).__init__()
        self.linear = nn.Linear(config.hidden_size * num_inputs, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, features):
        features = self.linear(features)
        features = gelu_new(features)
        features = self.LayerNorm(features)

        return features


class RobertaSpanHead(nn.Module):
    def __init__(self, config):
        super(RobertaSpanHead, self).__init__()
        self.layer_1 = SimpleLayer(config, 3)
        self.layer_2 = SimpleLayer(config, 1)

    def forward(self, features):
        features = self.layer_1(features)
        features = self.layer_2(features)

        return features


class RobertaForGappedText(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super(RobertaForGappedText, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.span_head = RobertaSpanHead(config)
        self.lm_head = RobertaLMHead(config)

        self.span_position_embeddings = nn.Embedding(300, config.hidden_size)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        mask_ids,
        span_features,
        mask_targets=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]

        mask_representations = sequence_output[mask_ids[:, 0], mask_ids[:, 1]]
        mask_scores = self.lm_head(mask_representations)

        start_representations = sequence_output[span_features[:, 0], span_features[:, 1]]
        end_representations = sequence_output[span_features[:, 0], span_features[:, 2]]
        mask_position_embeddings = self.span_position_embeddings(span_features[:, 3])
        span_features_new = self.span_head(torch.cat([start_representations, end_representations, mask_position_embeddings], dim=1))
        span_scores = self.lm_head(span_features_new)

        outputs = (span_scores, mask_scores) + outputs[2:]

        if mask_targets is not None:
            span_loss = F.cross_entropy(input=span_scores, target=mask_targets)
            lm_loss = F.cross_entropy(input=mask_scores, target=mask_targets)

            outputs = (span_loss, lm_loss) + outputs

        return outputs
