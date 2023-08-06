import torch.nn as nn


class SimpleNeuralNetwork(nn.Module):
    def __init__(
        self, input_dim, output_dim, layer1_dim=128, layer2_dim=64, act="relu"
    ):
        super(SimpleNeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_dim, layer1_dim)
        self.hidden_layer1 = nn.Linear(layer1_dim, layer2_dim)
        self.output_layer = nn.Linear(layer2_dim, output_dim)
        if act == "relu":
            self.act = nn.ReLU()
        if act == "tanh":
            self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.input_layer(x))
        x = self.act(self.hidden_layer1(x))
        x = self.output_layer(x)
        return x
