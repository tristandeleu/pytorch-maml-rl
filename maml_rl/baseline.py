import torch
import torch.nn as nn
from torch.autograd import Variable

def isnan(x):
    return x != x

class LinearFeatureBaseline(nn.Module):
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.linear = nn.Linear(self.feature_size, 1, bias=False)
        self.linear.weight.data.zero_()

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, episodes):
        ones = episodes.mask.unsqueeze(2)
        observations = episodes.observations * ones
        cum_sum = torch.cumsum(ones, dim=0) * ones
        al = cum_sum / 100.0

        return torch.cat([observations, observations ** 2,
            al, al ** 2, al ** 3, ones], dim=2)

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size
        featmat = self._feature(episodes).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        returns = episodes.returns.view(-1, 1)

        reg_coeff = self._reg_coeff
        eye = torch.eye(self.feature_size, dtype=torch.float32,
            device=self.linear.weight.device)
        for _ in range(5):
            coeffs, _ = torch.gels(
                torch.matmul(featmat.t(), returns),
                torch.matmul(featmat.t(), featmat) + reg_coeff * eye
            )
            if not isnan(self.linear.weight).any():
                break
            reg_coeff *= 10
        self.linear.weight.data = coeffs.data.t()

    def forward(self, episodes):
        features = self._feature(episodes)
        return self.linear(features)
