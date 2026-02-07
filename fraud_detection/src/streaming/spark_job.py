# src/streaming/spark_job.py
import os
import traceback
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    window, col, avg, sum as _sum, count,
    from_json, to_json, struct
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType, LongType
)


def start():
    # --- Config / env (override with env vars if needed) ---
    KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:29092")
    INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "transactions")
    OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "features")

    # Base dir for local testing - use an absolute path on Windows
    BASE_DIR = os.environ.get("STREAM_BASE_DIR", r"C:\tmp\fraud_stream")
    CHECKPOINT_FEATURES = os.path.join(BASE_DIR, "checkpoints", "features")
    CHECKPOINT_24H = os.path.join(BASE_DIR, "checkpoints", "agg_24h")
    CHECKPOINT_1H = os.path.join(BASE_DIR, "checkpoints", "agg_1h")
    AGG24_PATH = os.path.join(BASE_DIR, "agg_24h_parquet")

    # Ensure directories exist (local dev)
    os.makedirs(CHECKPOINT_FEATURES, exist_ok=True)
    os.makedirs(CHECKPOINT_24H, exist_ok=True)
    os.makedirs(CHECKPOINT_1H, exist_ok=True)
    os.makedirs(AGG24_PATH, exist_ok=True)

    # --- Spark session (include kafka package) ---
    # NOTE: prefer running with spark-submit and --packages; this config is a fallback
    spark = SparkSession.builder \
        .appName("fraud-features") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("INFO")

    # --- Schema for incoming JSON ---
    schema = StructType([
        StructField("tx_id", StringType(), True),
        StructField("user_id", IntegerType(), True),
        StructField("amount", DoubleType(), True),
        StructField("device", StringType(), True),
        StructField("ip_hash", IntegerType(), True),
        StructField("ts", TimestampType(), True),   # expects ISO-like timestamp in JSON
    ])

    # --- Read stream from Kafka ---
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
        .option("subscribe", INPUT_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # Cast value (bytes) -> string and parse JSON using from_json
    value_df = kafka_df.selectExpr("CAST(value AS STRING) AS json_str")
    parsed = value_df.select(from_json(col("json_str"), schema).alias("data")).select("data.*")

    # Use incoming ts as event time (assumes ts parses to TimestampType)
    parsed = parsed.withColumn("event_ts", col("ts"))

    # Watermark for late events (applies to subsequent aggregations)
    events = parsed.withWatermark("event_ts", "2 hours")

    # -------------------------
    # 24-hour sliding aggregate
    # -------------------------
    agg_24h = events.groupBy(
        col("user_id"),
        window(col("event_ts"), "24 hours", "1 hour")
    ).agg(
        count("*").alias("tx_count_24h"),
        _sum("amount").alias("amount_sum_24h")
    ).selectExpr(
        "user_id",
        "window.start as window_start",
        "window.end as window_end",
        "tx_count_24h",
        "amount_sum_24h"
    )

    agg_24h_query = agg_24h.writeStream \
        .format("parquet") \
        .option("path", AGG24_PATH) \
        .option("checkpointLocation", CHECKPOINT_24H) \
        .outputMode("append") \
        .trigger(processingTime="5 minutes") \
        .start()

    # -------------------------
    # 1-hour tumbling aggregate
    # -------------------------
    agg_1h = events.groupBy(
        col("user_id"),
        window(col("event_ts"), "1 hour")   # tumbling 1-hour windows
    ).agg(
        count("*").alias("tx_count_1h"),
        _sum("amount").alias("amount_sum_1h"),
        avg("amount").alias("amount_avg_1h")
    ).selectExpr(
        "user_id",
        "window.start as window_start",
        "window.end as window_end",
        "tx_count_1h",
        "amount_sum_1h",
        "amount_avg_1h"
    )

    def process_1h(batch_df, epoch_id):
        """
        ForeachBatch handler for 1-hour micro-batches.
        Writes joined output to Kafka.
        """
        try:
            if batch_df.rdd.isEmpty():
                return

            # Read persisted 24h data if present; otherwise create an empty DataFrame with explicit schema
            try:
                agg_24h_static = spark.read.parquet(AGG24_PATH)
            except Exception:
                empty_schema = StructType([
                    StructField("user_id", IntegerType(), True),
                    StructField("window_start", TimestampType(), True),
                    StructField("window_end", TimestampType(), True),
                    StructField("tx_count_24h", LongType(), True),
                    StructField("amount_sum_24h", DoubleType(), True),
                ])
                agg_24h_static = spark.createDataFrame([], schema=empty_schema)

            # Join on user_id and window_start
            joined = batch_df.join(
                agg_24h_static,
                on=["user_id", "window_start"],
                how="left"
            )

            # Prepare Kafka output: include key (user_id) so messages partition by user
            out = joined.selectExpr(
                "CAST(user_id AS STRING) AS key",
                "to_json(struct(*)) AS value"
            )

            # Batch-write to Kafka.
            out.write \
               .format("kafka") \
               .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
               .option("topic", OUTPUT_TOPIC) \
               .save()
        except Exception:
            # Log and re-raise for visibility
            traceback.print_exc()
            raise

    agg_1h_query = agg_1h.writeStream \
        .option("checkpointLocation", CHECKPOINT_1H) \
        .trigger(processingTime="1 minute") \
        .foreachBatch(process_1h) \
        .start()

    # --- Wait for streams (allow graceful shutdown of both) ---
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught: stopping streams...")
    finally:
        try:
            agg_24h_query.stop()
        except Exception:
            pass
        try:
            agg_1h_query.stop()
        except Exception:
            pass
        spark.stop()


if __name__ == '__main__':
    start()
