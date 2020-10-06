# -*- coding: utf-8 -*-
from layers.dynamic_rnn import DynamicLSTM
from layers.shap import Distribution_SHAP, Map_SHAP
import torch
import torch.nn as nn
import numpy as np


class SHAP_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(SHAP_LSTM, self).__init__()
        self.opt = opt
        self.embed_dim  = embedding_matrix.shape[-1]
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.shap = Distribution_SHAP(self.opt.max_seq_len, self.opt.polarities_dim, opt)
        self.map_shap = Map_SHAP(opt.embed_dim, opt.max_seq_len, opt)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs, label, weights, update=True):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]

        x = self.embed(text_raw_indices)

        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_idx = torch.tensor(torch.eq(text_raw_indices, aspect_indices[:, 0].reshape((-1, 1))),
                                  dtype=torch.float)
        aspect_pos_idx = [np.where(aspect_idx[i, :] == 1)[0] for i in range(len(aspect_idx))]

        if update:
            H_N, (h_n, _) = self.lstm(x, x_len)
            weights = self.shap(text_raw_indices, aspect_indices, label, H_N, weights, self.dense)

            out = self.dense(h_n[0])

            return out, weights

        else:
            if len(weights) != 0:
                x = self.map_shap(x, aspect_pos_idx, weights)
            else:
                pass
            _, (h_n, _) = self.lstm(x.to(self.opt.device), x_len)
            out = self.dense(h_n[0])

            return out

