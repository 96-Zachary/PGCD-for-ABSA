#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np
from layers.shap import Distribution_SHAP, Map_SHAP

class SHAP_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(SHAP_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.shap = Distribution_SHAP(opt.max_seq_len, opt.polarities_dim, opt)
        self.map_shap = Map_SHAP(opt.bert_dim, opt.max_seq_len, opt)

    def forward(self, inputs, label, weights, update=True):
        text_bert_indices, bert_segments_ids, aspect_indices = inputs[0], inputs[1], inputs[2]

        aspect_idx = torch.tensor(torch.eq(text_bert_indices, aspect_indices[:, 0].reshape((-1, 1))),
                                  dtype=torch.float)
        aspect_pos_idx = [np.where(aspect_idx[i, :] == 1)[0] for i in range(len(aspect_idx))]

        if update:
            sequence_output, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)

            weights =  self.shap(text_bert_indices, aspect_indices, label, sequence_output, weights, self.dense)

            pooled_output = self.dropout(pooled_output)
            logits = self.dense(pooled_output)

            return logits, weights

        else:
            _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)

            if len(weights) != 0:
                pooled_output = self.map_shap(pooled_output, aspect_pos_idx, weights)

            pooled_output = self.dropout(pooled_output)
            logits = self.dense(pooled_output)

            return logits
