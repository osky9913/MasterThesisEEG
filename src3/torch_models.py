from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


# 1. Simple Linear Model
class SimpleLinear(pl.LightningModule):
    def __init__(self, learning_rate):
        super(SimpleLinear, self).__init__()
        self.fc = nn.Linear(126, 8)
        self.learning_rate = learning_rate
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


# 2. Simple CNN
class SimpleCNN(pl.LightningModule):
    def __init__(self, learning_rate):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 126, 8)
        self.learning_rate = learning_rate
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = x.view(-1, 1, 126)  # Reshaping to [batch_size, 1, 126]
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


# 3. Simple RNN
class SimpleRNN(pl.LightningModule):
    def __init__(self, learning_rate):
        super(SimpleRNN, self).__init__()
        self.learning_rate = learning_rate
        self.loss = nn.BCEWithLogitsLoss()

        self.rnn = nn.RNN(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 8)


    def forward(self, x):
        x = x.view(-1, 126, 1)  # Reshaping to [batch_size, 126, 1]
        _, h_n = self.rnn(x)
        return self.fc(h_n.squeeze(0))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.size(0), -1)
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss)
        return loss


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
