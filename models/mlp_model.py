import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPModel, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 2048)
        self.dense_2 = nn.Linear(2048, num_classes)

    def forward(self, input):
        x = F.relu(self.dense_1(input))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.dense_2(x)
        output = F.log_softmax(x, dim=1)
        return output

