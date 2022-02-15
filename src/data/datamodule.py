
#%%
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch

# Helper function for one-hot encoding of mixed data
def one_hot_encode_mixed_data(X):
    # splitting into categorical and numerical columns 
    X_cat = X.loc[:, (X.dtypes=='object').values]
    X_num = X.drop(columns = X_cat.columns)
    
    enc = OneHotEncoder(drop='if_binary', sparse = False)
    X_cat_one_hot_fit = enc.fit_transform(X_cat)
    X_cat_one_hot = pd.DataFrame(
        X_cat_one_hot_fit, 
        columns=enc.get_feature_names(X_cat.columns), 
        index = X_cat.index)
        
    # Concatenating into a final data frame 
    X_final = pd.concat([X_num, X_cat_one_hot], axis = 1)

    return X_final 

#%% Helper function to define dataset for train loaders
class myData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.FloatTensor(X_data)
        self.y_data = torch.FloatTensor(y_data[:, None]) # Make vector into matrix
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


# Create data module to use in log reg and NN
class SanFranciscoDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.file_path = 'data/processed/sf_data_processed_2013_2018.csv'

        self.batch_size = 32 
        self.test_size = 0.2
        self.val_size = 0.2 # Relative to train val
        self.seed = 42

        self.id_var = 'id'
        #self.sens_vars = ['sex', 'age']
        self.y_var = 'is_violent'
        #self.dataset_name = 'Taiwanese Credit'

        self.load_raw_data()
        self.setup()

    def load_raw_data(self):
        self.raw_data = pd.read_csv(self.file_path)
    
    def setup(self, stage = None):
        #One hot encoding 
        X = self.raw_data.drop(columns = [self.y_var, self.id_var], axis = 1)
        X = one_hot_encode_mixed_data(X)
        y = self.raw_data[self.y_var].to_numpy()
        
        # Saving output and features for plNet
        self.n_obs = X.shape[0]
        self.n_features = X.shape[1]
        self.n_output = 1

        # Split data into train+validation and test
        all_idx = np.arange(self.n_obs) 
        train_val_idx, test_idx = train_test_split(
            all_idx, 
            test_size = self.test_size, 
            random_state = self.seed)
        X_train_val, y_train_val = X.iloc[train_val_idx], y[train_val_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

    
        # Split train+val into train and val
        X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
            X_train_val, 
            y_train_val, 
            train_val_idx,
            test_size = self.val_size, 
            random_state = self.seed)
       
        #Scaler to standardize for optimization step
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Saving test, train and val idx in self
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        if stage in (None, "fit"): 
            self.train_data = myData(X_train, y_train)
            self.val_data = myData(X_val, y_val)
        
        if stage in (None, "test"):
            self.test_data = myData(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

if __name__ == "__main__": 
    # testing if data module works 
    dm = SanFranciscoDataModule()
    dm.train_data.X_data
    dm.train_data.y_data