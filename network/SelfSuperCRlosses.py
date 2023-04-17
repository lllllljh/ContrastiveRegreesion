from __future__ import print_function
import torch.nn.functional as F
import torch
import torch.nn as nn


class SelfCRLoss(nn.Module):
    """Self vised Losses"""

    def __init__(self, temperature=2.0, base_temperature=2.0):
        super(SelfCRLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features):
        features = F.normalize(features, dim=1)
        device = torch.device('cuda')
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = features[:, 0]
        anchor_count = 1

        cos = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        cos_max, _ = torch.max(cos, dim=1, keepdim=True)
        logits = cos - cos_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
