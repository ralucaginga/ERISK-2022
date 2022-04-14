import torch
import torch.nn as nn
import numpy as np
import pdb

from transformers import BertForSequenceClassification, RobertaForSequenceClassification

from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertModel, BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput

# from textaugment import MIXUP
# mixup_tool = MIXUP(runs=2)

classifier_layer = lambda config: nn.Sequential(
    nn.Linear(config.hidden_size, 256),
    nn.Dropout(0.2),
    nn.GELU(),
    nn.Linear(256, config.num_labels)
)

class DepressedBert(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = classifier_layer(config)

class DepressedRoberta(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150 * 768, 2 * 768),
            nn.Dropout(0.2),
            nn.GELU(),
            
            nn.Linear(2 * 768, 256),
            nn.Dropout(0.2),
            nn.GELU(),
            
            nn.Linear(256, config.num_labels)
        )


class MixupBertModel(BertModel):
    def augment_embeddings(self, embedding_output, labels, alpha=0.2, gamma=0.5):
        bernoulli_coin = np.random.binomial(1, gamma, 1)[0]
        if bernoulli_coin == 0:
            return embedding_output, labels

        batch_size = embedding_output.size(0)
        device = embedding_output.device

        lam_vector = np.random.beta(alpha, alpha, batch_size)
        lam_tensor = torch.tensor(lam_vector).to(device)
        index = np.random.permutation(batch_size)

        mixed_embeddings = (embedding_output.T * lam_tensor).T + (embedding_output[index, :].T * (1.0 - lam_tensor)).T
        mixed_labels = (labels.T * lam_tensor).T + (labels[index].T * (1.0 - lam_tensor)).T
        return mixed_embeddings.float(), mixed_labels.float()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_output, labels = self.augment_embeddings(embedding_output, labels)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class MixupDepressedBert(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = MixupBertModel(config)
        self.classifier = classifier_layer(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

from datasets import EriskDataset
from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = MixupDepressedBert.from_pretrained("mental/mental-bert-base-uncased", num_labels=1).to(device)

train_dataset = EriskDataset('train')
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

for batch in train_dataloader:
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.float().to(device)

    output = model(input_ids, labels=labels)
    loss = output.loss
    break