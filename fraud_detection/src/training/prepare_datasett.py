import pandas as pd
import json
import boto3
import os
from datetime import datetime
import argparse

today_str = datetime.utcnow().strftime('%Y-%m-%d')

fraud_bucket = 'fraud-bucket2'
fraud_prefix = f"labels/{today_str}"

def load_records_minio(bucket_name,prefix):

    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin"
    )
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix= prefix,
    )
    objects = response.get('Contents',[])
    print(f"found {len(objects)} files")

    records = []
    for obj in objects:
        key = obj['Key']
        if not key.endswith('.json'):
            continue
        response = s3.get_object(Bucket=bucket_name,Key=key)
        body = response['Body'].read().decode('utf-8')
        record = json.loads(body)
        records.append(record)

    df = pd.DataFrame(records)

    return df
def join_with_label_delay(features:pd.DataFrame,labels:pd.DataFrame,label_delay_hours=24):
    assert 'tx_id' in features.columns and 'ts' in features.columns
    labels = labels.sort_values('label_ts').drop_duplicates('tx_id',keep='first').copy()
    merged = features.merge(labels[['tx_id','label_ts','confirmed_fraud']],on='tx_id',how='left')
    merged['ts'] = pd.to_datetime(merged['ts'],utc=True)
    merged['label_ts'] = pd.to_datetime(merged['label_ts'],utc=True)

    # merged['label_available_by'] = merged['ts'] + pd.Timedelta(hours=label_delay_hours)
    # merged['label_available'] = merged['label_ts'].notna() & (merged['label_ts'] <= merged['label_available_by'])
    # labeled = merged[merged['label_ts'].notna()].copy()
    labeled = merged.copy()
    labeled['confirmed_fraud'] = labeled['confirmed_fraud'].fillna(False)
    labeled['label'] = labeled['confirmed_fraud'].astype(int)

    return labeled

def save_parquet(df,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    df.to_parquet(path,index=False)
    print(f"Saved file to {path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--output", required=True)
#     args = parser.parse_args()

#     output_dir = args.output

#     fraud_df = load_records_minio(fraud_bucket, fraud_prefix)
#     features_df = load_records_minio("feature-bucket2", f"labels/{today_str}")

#     save_parquet(fraud_df, os.path.join(output_dir, "fraud.parquet"))
#     save_parquet(features_df, os.path.join(output_dir, "features.parquet"))


def main():
    path = "C:/Fraud-Dectection-Pipeline/fraud_detection/data/"
    fraud_df = load_records_minio(fraud_bucket, fraud_prefix)
    features_df = load_records_minio("feature-bucket2", f"labels/{today_str}")

    save_parquet(fraud_df, os.path.join(path, "fraud.parquet"))
    save_parquet(features_df, os.path.join(path, "features.parquet"))



if __name__ == "__main__":
    main()

