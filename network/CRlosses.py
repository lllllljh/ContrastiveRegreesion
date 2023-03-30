from __future__ import print_function
import torch
import torch.nn as nn


class SupCRLoss(nn.Module):
    """Supervised Losses"""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupCRLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        # generate mask and contrast feature
        device = torch.device('cuda')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute sim
        anchor_feature_norm = torch.norm(anchor_feature, dim=1, keepdim=True)
        contrast_feature_norm = torch.norm(contrast_feature, dim=1, keepdim=True)
        dist = anchor_feature_norm ** 2 + (contrast_feature_norm ** 2).transpose(0, 1) - 2 * torch.mm(anchor_feature,
                                                                                                      contrast_feature.transpose(
                                                                                                          0, 1))
        similarity = torch.exp(-dist)
        logits = torch.div(
            similarity,
            self.temperature)
        logits_exp = torch.exp(logits)

        # compute dis
        labels = labels.repeat(1 * anchor_count, batch_size * contrast_count)
        dis = torch.abs(labels - labels.transpose(0, 1)).float()

        # compute negative
        negative = torch.zeros(batch_size * contrast_count, batch_size * contrast_count, dtype=torch.float32).to(device)
        for i in range(batch_size * anchor_count):
            temp = dis.clone()
            for j in range(batch_size * contrast_count):
                column = temp[:, j]
                k = temp[i, j]
                column = torch.where(column >= k, 1, 0)
                temp[:, j] = column
            temp[:, i] = 0
            negative[i] = torch.matmul(logits_exp[i], temp)

        # compute loss
        for i in range(batch_size * anchor_count):
            negative[i][i] = 1
        for i in range(batch_size * anchor_count):
            logits[i][i] = 0
        log_negative = torch.log(negative)
        log_prob = logits - log_negative
        mean_log_prob_pos = log_prob.sum(1)/(batch_size * anchor_count - 1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
