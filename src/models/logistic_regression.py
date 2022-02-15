from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data.datamodule import SanFranciscoDataModule

def run_logistic_regression(dm):
    # Prepare for output data
    cols_to_keep = [dm.id_var]+[dm.y_var]
    # Remaining datasets
    output_data = (dm.raw_data[cols_to_keep]
        .iloc[dm.test_idx])
    
    # Prepare data
    X_train = dm.train_data.X_data
    y_train = dm.train_data.y_data
    test_idx = dm.test_idx
    X_test = dm.test_data.X_data

    # Fit model
    log_reg = LogisticRegression(
        solver = 'liblinear', 
        max_iter = 1000,
        random_state = 42)
    log_reg.fit(X_train, y_train)

    # Save predictions
    y_pred = log_reg.predict(X_test)
    y_prob = log_reg.predict_proba(X_test)[:,1] 
    output_data['log_reg_pred'] = y_pred
    output_data['log_reg_prob'] = y_prob

    # Print accuracy
    acc = accuracy_score(output_data.log_reg_pred, output_data[dm.y_var])
    print(f'Final accuracy score: {acc}')
    return output_data

# running logistic regression 
dm = SanFranciscoDataModule()
preds = run_logistic_regression(dm)
output_path = 'data/predictions/log_reg.csv'
preds.to_csv(output_path, index = False) 