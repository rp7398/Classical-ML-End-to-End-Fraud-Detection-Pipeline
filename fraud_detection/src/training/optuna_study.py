import argparse,json,os,uuid,math,warnings
from typing import Dict

import pandas as pd,numpy as np
import mlflow
import optuna
from optuna.integration import MLflowCallback,XGBoostPruningCallback

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

def load_parquet(path:str):
    df = pd.read_parquet(path)
    return df

def split_x_y(df:pd.DataFrame,label_col='label'):
    x = df.drop(columns=[label_col])
    y = df[label_col].astype(int)
    return x,y


def objective_factory(x_train,y_train,x_val,y_val,use_pruning=True):
    def objective(trial:optuna.Trial):
        params = {
            'verbosity':0,
            'objective':'binary:logistic',
            'tree_method':'hist',
            'n_estimators':trial.suggest_int('n_estimators',50,500),
            'max_depth':trial.suggest_int('max_depth',3,12),
            'learning_rate':trial.suggest_float('learning_rate',1e-3,0.3,log=True),
            'subsample':trial.suggest_float('subsample',0.5,1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree',0.5,1.0),
            'gamma':trial.suggest_float('gamma',0.0,5.0),
            'reg_alpha':trial.suggest_float('reg_alpha',0.0,5.0),
            'reg_lambda':trial.suggest_float('reg_lambda',0.0,5.0)
        }

        neg = int((y_train==0).sum())
        pos = int((y_train==1).sum())
        params['scale_pos_weight'] = max(1.0,neg/max(1,pos))


        with mlflow.start_run(nested=True) as run:
            mlflow.log_params(params)
            model = xgb.XGBClassifier(**params,use_label_encoder=False,eval_metric='auc',n_jobs=4)

            callbacks = []
            if use_pruning:
                pruning_cv = XGBoostPruningCallback(trial,'validation_0-auc')
                callbacks.append(pruning_cv)

            model.fit(x_train,y_train,
                      eval_set = [x_val,y_val],
                      early_stopping_rounds = 30,
                      verbose = False,
                      callbacks=callbacks if callbacks else None
                      )
            
            preds = model.predict_proba(x_val)[:,1]
            auc = roc_auc_score(y_val,preds)
            mlflow.log_metric('val_auc',float(auc))
        

        return 1.0 - float(auc)
    
    return objective


def run_study(
        train_path:str,
        val_path:str,
        mlflow_uri:str,
        experiment_name:str,
        n_trials:int = 50,
        storage:str = None,
        n_jobs:int = 1,
        study_name:str = None,
        seed:int = 42
):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    train_df = load_parquet(train_path)
    val_df = load_parquet(val_path)
    x_train,y_train = split_x_y(train_df)
    x_val,y_val = split_x_y(val_df)

    study = optuna.create_study(direction='minimize',
    study_name=study_name,storage=storage,load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=seed))

    mlflow_cv = MLflowCallback(tracking_uri=mlflow_uri,metric_name = 'val_auc')
    objective = objective_factory(x_train,y_train,x_val,y_val,use_pruning=True)

    study.optimize(objective,n_trials=n_trials,n_jobs=n_jobs,callbacks=[mlflow_cv])

    best = study.best_trial

    print('best_trial:',best.number,'value:',best.value)
    best_params = dict(best.params)
    final_model = xgb.XGBClassifier(**best_params,use_label_encoder = False,eval_metric='auc',n_jobs=4)

    x_combined = pd.concat([x_train,x_val],axis=0)
    y_combined = pd.concat([y_train,y_val],axis=0)
    final_model.fit(x_combined,y_combined,verbose=False)

    with mlflow.start_run(run_name='optuna_best_model') as run:
        mlflow.log_params(best_params)
        mlflow.log_metric('optuna_best_value',float(best.value))
        mlflow.log_param('optuna_best_trial',int(best.number))

        mlflow.sklearn.log_model(final_model,artifact_path='final_model_xgb')

        with open('best_params.json','w') as f:
            f.write(json.dumps(best_params,indent=2))
        mlflow.log_artifact('best_params.json')
    
    return study




