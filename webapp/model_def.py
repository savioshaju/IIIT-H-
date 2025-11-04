# model_def.py
import torch
import torch.nn as nn
import math

# ---------- Gradient Reversal Layer ----------
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def forward(self, x, lambd):
        return GradientReversal.apply(x, lambd)

# ---------- DANN Model ----------
class DANNModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=6, n_domains=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.domain_disc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_domains)
        )
        self.grl = GRL()

    def forward(self, x, grl_lambda=0.0):
        feat = self.encoder(x)
        class_logits = self.classifier(feat)
        rev_feat = self.grl(feat, grl_lambda)
        domain_logits = self.domain_disc(rev_feat)
        return class_logits, domain_logits
