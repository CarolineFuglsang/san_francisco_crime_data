#%% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger('lightning').setLevel(logging.ERROR)

## Helper functions for running NN
#%% Pytorch network
class Net(nn.Module):
    def __init__(self, num_features, num_hidden_list, num_output, p_dropout = 0):
        super(Net, self).__init__()
        n_hidden_layers = len(num_hidden_list)
        self.layers = []
        
        input_dim = num_features
        for i in range(n_hidden_layers):
            output_dim = num_hidden_list[i]
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.layers.append(nn.Dropout(p = p_dropout))
            self.layers.append(nn.ReLU())
            input_dim = output_dim
        
        # Last layer (without activation function)
        self.layers.append(nn.Linear(num_hidden_list[-1], num_output))
        self.layers.append(nn.Dropout(p = p_dropout))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        x = torch.sigmoid(x) 

        return x

#%% Pytorh Lightning network
class BinaryClassificationTask(pl.LightningModule):
    def __init__(self, model, lr = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"train_loss": loss, 'train_acc': acc}
        self.log_dict(metrics, on_epoch = True, on_step = False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss, 'val_acc': acc}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss, 'test_acc': acc}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy(y_hat, y)
        y_hat_binary = (y_hat >= 0.5)
        n_batch = y.size(0)
        accuracy = (y_hat_binary == y).sum().item()/n_batch
        return loss, accuracy

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.lr)

# Helper for easy printing of elapsed time
def print_timing(t0, t1, text = 'Minutes elapsed:'):
    n_mins = (t1-t0)/60
    print(f'{text} {n_mins:.2f} mins')
