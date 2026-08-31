"""PySpark transformations for MLApp."""
from pyspark.sql import DataFrame


def clean_data(df):
    # Add null check for customer_id
    df = df.dropna(subset=['customer_id'])
    df = df.filter(df['amount'] > 0)
    return df


def calculate_totals(df: DataFrame) -> DataFrame:
    """Calculate total spend per customer."""
    from pyspark.sql import functions as F
    return df.groupBy("customer_id").agg(
        F.sum("amount").alias("total_spend")
    )


def segment_customers(df):
    # Scenario 4: accidental change on main
    """RFM-based customer segmentation for MLAPP-1234."""
    from pyspark.sql import functions as F
    return df.withColumn(
        "segment",
        F.when(F.col("total_spend").isNull(), "Bronze")
        .when(F.col("total_spend") > 10000, "Premium")
        .when(F.col("total_spend") > 5000, "Gold")
        .when(F.col("total_spend") > 1000, "Silver")
        .otherwise("Bronze")
    )