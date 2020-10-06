#-*-coding:utf-8-*-

from layers.dynamic_rnn import DynamicLSTM
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SHAP_AOA(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(SHAP_AOA, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.shapy_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.shap_dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.dense = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs, epoch, targets, weights={1:1}):
        text_raw_indices = inputs[0] # batch_size x seq_len
        aspect_indices = inputs[1] # batch_size x seq_len
        ctx_len = torch.sum(text_raw_indices != 0, dim=1)
        ctx_len_max = torch.max(ctx_len)
        asp_len = torch.sum(aspect_indices != 0, dim=1)

        ctx = self.embed(text_raw_indices) # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices) # batch_size x seq_len x embed_dim
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len) #  batch_size x (ctx) seq_len x 2*hidden_dim
        asp_out, (_, _) = self.asp_lstm(asp, asp_len) # batch_size x (asp) seq_len x 2*hidden_dim
        interaction_mat = torch.matmul(ctx_out, torch.transpose(asp_out, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(interaction_mat, dim=1) # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2) # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True) # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # batch_size x (ctx) seq_len x 1

        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(-1) # batch_size x 2*hidden_dim


        if epoch <= 4:
            out = self.dense(weighted_sum)  # batch_size x polarity_dim

            aspect_idx = torch.tensor(torch.eq(text_raw_indices, aspect_indices[:, 0].reshape((-1, 1))),
                                      dtype=torch.long)
            pos_idx = [np.where(aspect_idx[i, :] == 1)[0] for i in range(len(aspect_idx))]
            context_idx = torch.LongTensor(1 - aspect_idx)

            i = 0
            for tmp in ctx_len:
                shap_out = torch.zeros(out[i].shape, dtype=torch.float)
                shap_out[targets[i]] = 1

                pred = (out - shap_out)[:, targets[i]].detach()
                tmp_weight = torch.zeros((1, self.opt.max_seq_len), dtype=torch.float, requires_grad=False)
                pred = F.softmax(pred[:tmp])
                if len(pred) <= 40:
                    tmp_weight[0, :len(pred)] = pred
                    for pos in pos_idx[i]:
                        if pos in weights:
                            weights[pos] = (tmp_weight + weights[pos]) / 2
                        else:
                            weights[pos] = torch.FloatTensor(tmp_weight)
                else:
                    for pos in pos_idx[i]:
                        if pos in weights:
                            weights[pos] = (tmp_weight + weights[pos]) / 2
                        else:
                            weights[pos] = torch.ones((1, self.opt.max_seq_len),
                                                      dtype=torch.float) / self.opt.max_seq_len
                i += 1
        else:
            aspect_idx = torch.tensor(torch.eq(text_raw_indices, aspect_indices[:, 0].reshape((-1, 1))),
                                      dtype=torch.long)
            pos_idx = [np.where(aspect_idx[i, :] == 1)[0] for i in range(len(aspect_idx))]

            shapy_matrix = torch.stack([weights[j[0]] for j in pos_idx], dim=0)

            gamma = gamma + 0.1*torch.mul(gamma, shapy_matrix.permute(0,2,1)[:, :ctx_len_max, :])
            weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(-1)  # batch_size x 2*hidden_dim

            out = self.dense(weighted_sum) # batch_size x polarity_dim

        return out, weights
