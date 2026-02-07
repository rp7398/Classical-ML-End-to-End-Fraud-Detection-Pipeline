import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parents[2]  
# cwd = fraud_detection/src/training
# parents[2] = Fraud-Detection-Pipeline

sys.path.insert(0, str(PROJECT_ROOT))

import argparse,os,json
from pathlib import Path
from typing import Tuple,Dict

import numpy as np,pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder,FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


import joblib
import mlflow

import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader,TensorDataset

from fraud_detection.src.utils.common import cast_to_str
from fraud_detection.src.training.evaluator import business_loss,evaluate_model,find_best_threshold

LABEL_COL = 'label'

def read_data(path:str):
    df = pd.read_parquet(path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"{LABEL_COL} not found in {path}")
    
    return df

def split_x_y(df:pd.DataFrame):
    df = df.copy()
    y = df[LABEL_COL].astype(int)
    x = df.drop(columns=[LABEL_COL])

    return x,y
# def cast_to_str(df):
#     return df.astype(str)


def Build_preprocessor(
        X: pd.DataFrame,
        endcoder_type: str = 'ordinal',
        onehot_drop_first: bool = False
):
    num_cols = X.select_dtypes(include=['number']).columns.to_list()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    if endcoder_type == 'onehot':
        drop_arg = 'first' if onehot_drop_first else None
        cat_encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            drop=drop_arg
        )
    elif endcoder_type == 'ordinal':
        cat_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
    else:
        raise ValueError("Encoder Type must be 'ordinal' or 'onehot'")

    cat_pipeline = Pipeline([
        ('as_str', FunctionTransformer(cast_to_str, validate=False)),
        ('encoder', cat_encoder)
    ])

    transformers = []
    if num_cols:
        transformers.append(('num', num_pipeline, num_cols))
    if cat_cols:
        transformers.append(('cat', cat_pipeline, cat_cols))

    if not transformers:
        raise ValueError("No numeric or categorical columns found")

    col_transformer = ColumnTransformer(
        transformers,
        remainder='drop',
        sparse_threshold=0.0
    )

    preprocesor_pipeline = Pipeline([
        ('preprocessor', col_transformer)
    ])

    preprocesor_pipeline.fit(X)

    try:
        feature_names = list(
            preprocesor_pipeline
            .named_steps['preprocessor']
            .get_feature_names_out()
        )
    except Exception:
        feature_names = []

    def transform(df: pd.DataFrame):
        missing = [c for c in num_cols + cat_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        arr = preprocesor_pipeline.transform(df)
        return np.asarray(arr)

    preprocessor_meta = {
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'preprocessor': preprocesor_pipeline,
        'feature_names': feature_names,
        'encoder_type': endcoder_type
    }

    return preprocessor_meta, transform









def train_xgboost(
    x_train,
    y_train,
    x_val,
    y_val,
    preprocess_meta,
    params: Dict
):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = max(1.0, neg / max(1, pos))

    clf = xgb.XGBClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        n_jobs=params.get("n_jobs", -1),
        early_stopping_rounds=30
    )

    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False
    )

    pred_proba = clf.predict_proba(x_val)[:, 1]
    pred = (pred_proba >= params.get("threshold", 0.5)).astype(int)

    metrics = evaluate_preds(y_val, pred_proba, pred)


    joblib.dump(clf, "C:/Fraud-Dectection-Pipeline/fraud_detection/models/xgb_model.joblib")
    mlflow.log_artifact("C:/Fraud-Dectection-Pipeline/fraud_detection/models/xgb_model.joblib",
                        artifact_path="model")

    joblib.dump(preprocess_meta,"C:/Fraud-Dectection-Pipeline/fraud_detection/models/preprocess_meta.joblib")
    mlflow.log_artifact("C:/Fraud-Dectection-Pipeline/fraud_detection/models/preprocess_meta.joblib",
                        artifact_path="preprocess")
    mlflow.sklearn.log_model(clf,artifact_path="sklearn_model")

    return metrics,clf




def evaluate_preds(y_true,y_proba,y_pred):
    auc = roc_auc_score(y_true,y_proba) if len(np.unique(y_true)) > 1 else 0.5
    precision = precision_score(y_true,y_pred,zero_division=0)
    recall = recall_score(y_true,y_pred,zero_division=0)
    f1 = f1_score(y_true,y_pred,zero_division=0)
    return {'auc':float(auc),'precision':float(precision),'recall':float(recall),'f1':float(f1)}

def main():
    import os
    os.environ["MLFLOW_ENABLE_MODEL_REGISTRY"] = "false"

    fraud_path = 'C:/Fraud-Dectection-Pipeline/fraud_detection/data/fraud.parquet'
    fraud_df = pd.read_parquet(fraud_path)
    features_path = 'C:/Fraud-Dectection-Pipeline/fraud_detection/data/features.parquet'
    features_df = pd.read_parquet(features_path)

    features_df = features_df.rename(columns={'ts':'label_ts'})

    features_df['confirmed_fraud'] = bool(False)

    common_cols = [
        "tx_id",
        "user_id",
        "amount",
        "device",
        "ip_hash",
        "metadata",
        "confirmed_fraud",
        "label_ts"
    ]

    fraud_df = fraud_df[common_cols]
    features_df = features_df[common_cols]

    final_df = pd.concat([features_df,fraud_df],ignore_index=True,axis=0)

    final_df = final_df.sample(frac=1.0,random_state=42).reset_index(drop=True)

    x = final_df.drop(columns=['tx_id','user_id','confirmed_fraud'])
    y = final_df['confirmed_fraud'].astype(int)

    from sklearn.model_selection import train_test_split
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    amount_val = x_val['amount'].values

    preprocess_meta,preprocess_func = Build_preprocessor(x_train,endcoder_type='ordinal',onehot_drop_first=False)
    x_train = preprocess_func(x_train)
    x_val = preprocess_func(x_val)

    params = {
            "model": 'xgboost',
            "threshold": 0.5,
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "epochs": 20 ,
            "batch_size": 1024,
            "use_focal": 'store_true'
        }
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud_detection_experiment")


    import os

    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"



    with mlflow.start_run(run_name='xgboost_baseline') as run:
        metrics,clf = train_xgboost(x_train,y_train,x_val,y_val,preprocess_meta,params)
        for k,v in metrics.items():
            mlflow.log_metric(k,v)
        eval_df = pd.DataFrame([metrics])
        eval_df.to_csv('C:/Fraud-Dectection-Pipeline/fraud_detection/data/eval_metrics.csv',index=False)
        mlflow.log_artifact('C:/Fraud-Dectection-Pipeline/fraud_detection/data/eval_metrics.csv',
                            artifact_path='evaluation')
        y_proba = clf.predict_proba(x_val)[:,1]
        best_t,best_loss = find_best_threshold(y_val,y_proba,amount_val)
        metrics_eco = evaluate_model(y_val,y_proba,amount_val,best_t)
        mlflow.log_metric('val_auc',metrics_eco['auc'])
        mlflow.log_metric('business_loss',metrics_eco['business_loss'])
        mlflow.log_metric('best_threshold',best_t)
    print("Training complete.")
    print("Metrics:",metrics)
    
if __name__ == "__main__":
    main()
