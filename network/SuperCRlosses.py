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
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute sim
        dist = torch.cdist(anchor_feature, contrast_feature, p=2)

        logits = torch.div(
            -1 * dist,
            self.temperature)
        logits_exp = torch.exp(logits)

        # compute dis
        labels = labels.repeat(1 * anchor_count, batch_size * contrast_count)
        dis = torch.abs(labels - labels.transpose(0, 1)).float()

        # compute negative
        negative = torch.zeros(batch_size * contrast_count, batch_size * contrast_count, dtype=torch.float32).to(device)
        for i in range(batch_size * anchor_count):
            temp = torch.zeros(batch_size * contrast_count, batch_size * contrast_count, dtype=torch.float32).to(device)
            column = dis[:, i].clone()
            for j in range(batch_size * contrast_count):
                k = dis[i, j]
                temp[:, j] = torch.where(column >= k, 1, 0)
            temp[:, i] = 0
            temp[i] = 0
            negative[i] = torch.matmul(logits_exp[i], temp)

        # compute loss
        for i in range(batch_size * anchor_count):
            negative[i][i] = 1
        for i in range(batch_size * anchor_count):
            logits[i][i] = 0
        log_negative = torch.log(negative)
        log_prob = logits - log_negative
        mean_log_prob_pos = log_prob.sum(1) / (batch_size * anchor_count - 1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
