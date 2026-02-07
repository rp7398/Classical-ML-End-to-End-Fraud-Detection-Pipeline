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


def Build_preprocessor(
        X:pd.DataFrame,
        endcoder_type:str = 'ordinal',
        onehot_drop_first:bool = False
):
    num_cols = X.select_dtypes(include=['number']).columns.to_list()
    cat_cols = X.select_dtypes(include=['object','category']).columns.to_list()

    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ])


    if endcoder_type == 'onehot':
        drop_arg = 'first' if onehot_drop_first else None
        cat_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False,drop=drop_arg)
    
    elif endcoder_type == 'ordinal':
        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    else:
        raise ValueError(f"Encoder Type must be ordinal or onehot")
    
    cat_pipeline = Pipeline([
        ('as_str',FunctionTransformer(lambda df:df.astype(str),validate=False)),
        ('encoder',cat_encoder)
    ])

    transformers = []
    if num_cols:
        transformers.append(('num',num_pipeline,num_cols))
    if cat_cols:
        transformers.append(('cat',cat_pipeline,cat_cols))

    
    if not transformers:
        raise ValueError('No numeric or categorical column found in the training dataframe')
    
    col_transformer = ColumnTransformer(transformers,remainder='drop',sparse_threshold=0.0)
    preprocesor_pipeline = Pipeline(['preprocessor',col_transformer])
    preprocesor_pipeline.fit(X)

    try:
        feature_names = preprocesor_pipeline.named_steps['preprocessor'].get_feature_names_out()
        feature_names = list(feature_names)
    except Exception:
        feature_names = []
    

    def transform(df:pd.DataFrame):
        missing_num = [c for c in num_cols if c not in df.columns]
        missing_cat = [c for c in cat_cols if c not in df.columns]

        missing = missing_cat + missing_num
        if missing:
            raise ValueError(f"Input Dataframe is missing required columns: {missing}."
                             f"Expected numeric:{num_cols}, categorical:{cat_cols}")
        
        arr = preprocesor_pipeline.transform(df)
        if not isinstance(arr,np.ndarray):
            arr = np.asarray(arr)

        return arr
    
    preprocessor_meta = {
        'num_cols':num_cols,
        'cat_cols':cat_cols,
        'preprocessor':preprocesor_pipeline,
        'feature_names':feature_names,
        'encoder_type':endcoder_type
    }

    return preprocessor_meta,transform


class SimpleTabularNet(nn.Module):
    def __init__(self,input_dim,hidden=128,droupout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden),
            nn.ReLU(),
            nn.Dropout(droupout),
            nn.Linear(hidden,hidden//2),
            nn.ReLU(),
            nn.Dropout(droupout),
            nn.Linear(hidden//2,1)
        )

    def forward(self,x):
        return self.net(x).squeeze(-1)
    

def focal_loss_sigmoid(logits,targets,alpha=0.25,gamma=2.0,reduction='mean'):
    preds = torch.sigmoid(logits)
    targets = targets.float()
    pt = preds * targets + (1 - preds) * (1 - targets)
    w = alpha * (1 - pt) ** gamma
    bce = nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    loss = bce * w

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    


def train_logistic(x_train,y_train,x_val,y_val,preprocess_meta,params:Dict,mlflow_client:mlflow.client=None):
    
    class_weight = 'balanced' if params.get('class_weight','balanced') == 'balanced' else None
    clf = LogisticRegression(max_iter=1000,class_weight=class_weight,solver='saga')
    clf.fit(x_train,y_train)
    pred_proba = clf.predict_proba(x_val)[:,1]
    preds = (pred_proba == params.get('threshold',0.5)).astype(int)
    metrics = evaluate_preds(y_val,pred_proba,preds)
    mlflow.sklearn.log_model(clf,'model')
    joblib.dump(preprocess_meta,'preprocess_meta.joblib')
    mlflow.log_artifact('preprocess_meta.joblib')

    return metrics


def train_xgboost(x_train,y_train,x_val,y_val,preprocess_meta,params:Dict):
    neg = (y_train == 0).sum()
    pos = (x_train == 0).sum()
    scale_pos_weight = max(1.0,neg/max(1,pos))

    clf = xgb.XGBClassifier(
        n_estimators = int(params.get('n_estimators',200)),
        max_depth = int(params.get('max_depth',6)),
        learning_rate = float(params.get('learning_rate',0.1)),
        subsample = float(params.get('subsample',0.8)),
        colsample_bytree = float(params.get('colsample_bytree',0.8)),
        use_label_encoder = False,
        eval_metric = 'auc',
        scale_pos_weight=scale_pos_weight,
        n_jobs = params.get('n_jobs',-1)
    )

    clf.fit(x_train,y_train,eval_set=[(x_val,y_val)],early_stopping_rounds=30)
    pred_proba = clf.predict_proba(x_val)[:,1]
    pred = (pred_proba >= params.get('threshold',0.5)).astype(int)
    metrics = evaluate_preds(y_val,pred_proba,pred)

    mlflow.sklearn.log_model(clf,'model')
    joblib.dump(preprocess_meta,'preprocess_meta.joblib')
    mlflow.log_artifact('preprocess_meta.joblib')
    return metrics

def train_torch(x_tarin,y_train,x_val,y_val,preprocess_meta,params:Dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imput_dim = x_tarin.shape[1]
    model = SimpleTabularNet(imput_dim,hidden=int(params.get('hidden',128)),droupout=params.get('droupout',0.2))
    model.to(device)

    batch_size = int(params.get('batch_size',1024))
    train_ds = TensorDataset(torch.from_numpy(x_tarin.astype(np.float32)),
                             torch.from_numpy(y_train.values.astype(np.float32)))
    test_ds = TensorDataset(torch.from_numpy(x_val.astype(np.float32)),
                            torch.from_numpy(y_val.values.astype(np.float32)))
    
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=False)

    pos = max(1,int(y_train.sum()))
    neg = max(1,int(len(y_train)-pos))
    pos_weight = torch.tensor([neg/pos],dtype=torch.float32).to(device)
    use_focal = params.get('use_focal',False)

    optimizer = optim.Adam(model.parameters(),lr=float(params.get('lr',1e-3)))
    epochs = int(params.get('epoches',20))
    best_auc = 0.0
    early_stop_rounds = params.get('early_stop',5)
    rounds = 0


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb,yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            if use_focal:
                loss = focal_loss_sigmoid(logits,yb,alpha=params.get('focal_alpha',0.25),
                                          gamma=float(params.get('focal_gamma',2.0)))
            else:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = criterion(logits,yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        
        model.eval()
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for xb,yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(yb.numpy())
        
        preds_proba = np.concatenate(all_probs,axis=0)
        targets = np.concatenate(all_targets,axis=0)
        auc = roc_auc_score(targets,preds_proba)

        if auc > best_auc + 1e-5:
            best_auc = auc
            rounds = 0
            torch.save(model.state_dict(),'best_model.pth')
        else:
            rounds += 1
            if rounds >= early_stop_rounds:
                break
        
    
    model.load_state_dict(torch.load('best_model.pth'))
    final_probs = preds_proba
    final_preds = (preds_proba >= params.get('threshold',0.5)).astype(int)
    metrics = evaluate_preds(y_val,final_probs,final_preds)

    mlflow.pytorch.log_model(model,'model')
    joblib.dump(preprocess_meta,'preprocess_meta.joblib')
    mlflow.log_artifact('preprocess_meta.joblib')

    return metrics

def evaluate_preds(y_true,y_proba,y_pred):
    auc = roc_auc_score(y_true,y_proba) if len(np.unique(y_true)) > 1 else 0.5
    precision = precision_score(y_true,y_pred,zero_division=0)
    recall = recall_score(y_true,y_pred,zero_division=0)
    f1 = f1_score(y_true,y_pred,zero_division=0)
    return {'auc':float(auc),'precision':float(precision),'recall':float(recall),'f1':float(f1)}



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train',required=True,help='train parquet path')
    p.add_argument('--val',required=True,help='val argument path')
    p.add_argument('--mlflow',default="http://localhost:5000",help='mlflow tracking uri')
    p.add_argument('--experiment',default='fraud detection',help='MLflow experiment')
    p.add_argument('--model',choices=['logistic','xgboost','torch'],default='xgboost')
    p.add_argument('--output-dir',default='./outputs',help='local outputs')
    p.add_argument('--n-estimators',type=int,default=200)
    p.add_argument('--max-depth',type=int,default=6)
    p.add_argument('--learning-rate',type=float,default=0.1)
    p.add_argument('--epoches',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--threshold',type=float,default=0.5)
    p.add_argument('--use-focal',action='store_true')

    


    args = p.parse_args()
    mlflow.set_tracking_uri(args.mlflow)
    mlflow.set_experiment(args.set_experiment)
    os.makedirs(args.output_dir,exist_ok=True)

    train_df = read_data(args.train)
    val_df = read_data(args.val)

    x_train_df,y_train = split_x_y(train_df)
    x_val_df,y_val = split_x_y(val_df)

    preprocessor_meta,transform_fn = Build_preprocessor(x_train_df)
    x_train = transform_fn(x_train_df)
    x_val = transform_fn(x_val_df)

    params = {
        "model": args.model,
        "threshold": args.threshold,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "use_focal": args.use_focal
    }

    with mlflow.start_run(run_name=f"{args.model}_baseline"):
        mlflow.log_params(params)
        if args.model == 'logistic':
            metrics = train_logistic(x_train,y_train,x_val,y_val,preprocessor_meta,params)
        elif args.model == 'xgboost':
            metrics = train_xgboost(x_train,y_train,x_val,y_val,preprocessor_meta,params)
        elif args.model == 'torch':
            metrics = train_torch(x_train,y_train,x_val,y_val,preprocessor_meta,params)
        else:
            raise ValueError('Unknown model')
        
        for k,v in metrics.items():
            mlflow.log_metric(k,v)

        eval_df = pd.DataFrame([metrics])
        eval_df.to_csv('metrics.csv',index=False)
        mlflow.log_artifact('metrics.csv')

    print('Done. Metrics:',metrics)
    print('Artifact Saved to Mlflow at',args.mlflow)


if __name__ == '__main__':
    main()








