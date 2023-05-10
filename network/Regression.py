import torch
import torch.nn as nn


class Regression(nn.Module):

    def __init__(self, sex_input_size=1, sex_output_size=8, input_size=512, output_size=1):
        super(Regression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sex_input_size = sex_input_size
        self.sex_output_size = sex_output_size
        self.linear1 = nn.Linear(self.input_size + sex_output_size, self.input_size)
        self.linear2 = nn.Linear(self.input_size, self.output_size)
        self.linear3 = nn.Linear(self.sex_input_size, self.sex_output_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x, sex):
        sex = self.linear3(sex)
        sex = self.relu(sex)
        x = torch.cat([x, sex], dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
