# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Distribution_SHAP(nn.Module):
    def __init__(self, max_len, outs_dim, opt):
        super(Distribution_SHAP, self).__init__()
        self.max_len = max_len
        self.outs_dim = outs_dim
        self.opt = opt

    def forward(self, text_raw_indices, aspect_indices, label, H_N, weights, dense):
        if 'bert' in self.opt.model_name:
            # x_len = torch.sum(text_raw_indices != 0, dim=-1)
            aspect_idx = torch.tensor(torch.eq(text_raw_indices, aspect_indices[:, 1].reshape((-1, 1))),
                                      dtype=torch.float)
        else:
            # x_len = torch.sum(text_raw_indices != 0, dim=-1)
            aspect_idx = torch.tensor(torch.eq(text_raw_indices, aspect_indices[:, 0].reshape((-1, 1))),
                                      dtype=torch.float)
        aspect_pos_idx = [np.where(aspect_idx[i, :] == 1)[0] for i in range(len(aspect_idx))]

        aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float)
        context_idx = torch.FloatTensor(1 - aspect_idx)

        # context_idx = [batch_size, senten_len]
        # update context_idx and aspect_idx
        for i in range(len(aspect_pos_idx)):
            length = aspect_len[i]
            if int(length) != 1:
                for tmp_pos in aspect_pos_idx[i]:
                    start_idx = int(tmp_pos)
                    context_idx[i, start_idx:start_idx + int(aspect_len[i])] = 0.0
                    aspect_idx[i, start_idx:start_idx + int(aspect_len[i])] = 1

        context_matrix = context_idx.unsqueeze(-1).repeat(1, 1, H_N.shape[-1])
        context_text = torch.mul(context_matrix[:H_N.shape[0], :H_N.shape[1], :H_N.shape[2]], H_N)

        onehot_label = torch.zeros(context_text.shape[0], self.outs_dim).scatter_(1, label.reshape((-1, 1)), 1)
        onehot_label = onehot_label.unsqueeze(1).repeat(1, context_text.shape[1], 1)

        context_outs = dense(context_text.to(self.opt.device))
        cont_value = onehot_label.cpu() - context_outs.cpu()
        idx = torch.tensor(label).long().view(len(label), 1, 1).repeat(1, cont_value.shape[1], 1)
        cont_value = cont_value.gather(-1, idx).squeeze()

        for j in range(cont_value.shape[0]):
            pos = aspect_pos_idx[j]
            pos_cont = cont_value[j, :]
            tmp_cont = torch.zeros(self.max_len).float()
            tmp_cont[:len(pos_cont)] = F.softmax(pos_cont)

            pos = pos[int(len(pos) / 2)] if len(pos) != 0 else None
            if pos in weights:
                weights[pos] = (weights[pos] + tmp_cont.detach().numpy()) / 2
            else:
                weights[pos] = tmp_cont.detach().numpy()

        return weights


class Map_SHAP(nn.Module):
    def __init__(self, embed_dim, max_len, opt):
        super(Map_SHAP, self).__init__()
        self.opt = opt
        self.max_len = max_len
        self.embed_dim = embed_dim

    def forward(self, x, aspect_pos_idx, weights):
        j = 0
        for pos in aspect_pos_idx:
            int_pos = pos[int(len(pos) / 2)] if len(pos) != 0 else None
            weight = torch.tensor(weights[int_pos].reshape((1, -1))) if int_pos in weights else \
                torch.zeros((1, self.max_len))

            if j == 0:
                weight_matrix = weight.reshape((1, -1))
            else:
                weight_matrix = torch.cat((weight_matrix, weight), dim=0)
            j += 1

        weight_matrix = F.softmax(1 - weight_matrix.unsqueeze(-1).repeat(1, 1, self.embed_dim), dim=1)

        x = weight_matrix.to(self.opt.device) + x

        return x
