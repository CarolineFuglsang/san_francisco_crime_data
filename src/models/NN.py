#%% Imports
import time
import numpy as np
import platform
import os
import pandas as pd

import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


from src.models.helper_functions_for_NN import (Net,\
    BinaryClassificationTask, print_timing)
from src.data.datamodule import SanFranciscoDataModule


#%%
if __name__ == "__main__":
        
    ##### DEFINITIONS #######

    model_name = 'NN'
    model_path = f'models/checkpoints_from_trainer/{model_name}'
    output_path = 'data/predictions/NN_pred.csv'
    param_path = 'data/predictions/NN_pred_hyperparams.csv'

    fast_dev_run = False
    
    # All defined variables below must be included into hyper_dict
    max_epochs = 10
    lr = 0.0001
    num_hidden_list = [3, 6]
    p_dropout = 0.2
    early_stopping = False
    early_stopping_patience = 3    
    
    hyper_dict = {
        'max_epochs': max_epochs,
        'lr': lr,
        'early_stopping': early_stopping,
        'early_stopping_patience': early_stopping_patience,
        }

    model_checkpoint_callback = ModelCheckpoint(
        dirpath = model_path, 
        filename=f"{model_name}-"+"{epoch}-{step}-{val_loss:.2f}",
        save_top_k = 1, 
        save_last=True, 
        monitor='val_loss', 
        mode='min'   
    )

    ##### Setup #########
    if torch.cuda.is_available():
        GPU = 1
    else:
        GPU = None
    if platform.system() == 'Linux':
        n_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(n_avail_cpus-1, 8)
    else:
        num_workers = 0
    print(f"num workers are {num_workers}")

    pl.seed_everything(42)
    t0 = time.time()

    print('--- Initializing model and datamodule ---') 
    dm = SanFranciscoDataModule()

    net = Net(
        num_features = dm.n_features, 
        num_hidden_list = num_hidden_list,
        num_output = dm.n_output, 
        p_dropout = p_dropout)
    pl_model = BinaryClassificationTask(model = net, lr = lr)


    print('--- Setup training ---')
    logger = TensorBoardLogger(
        save_dir = 'models/lightning_logs', 
        name = model_name)
    logger.log_hyperparams(hyper_dict)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    callbacks = [model_checkpoint_callback, lr_monitor_callback]
    if early_stopping:
        early_stopping = EarlyStopping(
            'val_loss', 
            patience = early_stopping_patience)
        callbacks.append(early_stopping)

    print('--- Training model ---')
    trainer = pl.Trainer(
        fast_dev_run = fast_dev_run,
        max_epochs = max_epochs,
        deterministic = True,
        enable_checkpointing = True,
        callbacks = callbacks,
        progress_bar_refresh_rate = 0,
        gpus = GPU,
        logger = logger)
    
    trainer.fit(pl_model, dm)
    
    print('--- Testing and making predictions using best model ---')
    cols_to_keep = [dm.id_var, dm.y_var]
    output_data = (dm.raw_data[cols_to_keep]
        .iloc[dm.test_idx]
        .assign(
            nn_prob = np.nan,
            nn_pred = np.nan
        ))

    pl_model.model.eval()
    nn_prob = (pl_model.model
        .forward(dm.test_data.X_data)
        .detach().numpy().squeeze())
    assert(nn_prob.shape[0] == output_data.shape[0])
    output_data.nn_prob = nn_prob
    output_data.nn_pred = nn_prob >= 0.5
    
    acc = accuracy_score(output_data.nn_pred, output_data[dm.y_var])
    print(f'Final accuracy score: {acc}')

    print('--- Saving predictions and Hyper Parameters ---')
    params = {'lr':lr, 'num_hidden_list': num_hidden_list, 'p_dropout': p_dropout}
    output_data.to_csv(output_path, index = False)
    params_df = pd.DataFrame([params], columns = params.keys())
    params_df.to_csv(param_path, index = False)


    ### FINISHING UP ####
    t1 = time.time()
    print_timing(t0, t1, text = 'Total time to run script:')

