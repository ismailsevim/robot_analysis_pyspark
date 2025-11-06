import pytest
from pyspark.sql import SparkSession
import os


@pytest.fixture(scope="module")
def spark():
    """Create and tear down a Spark session for testing."""
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("pytest-pyspark")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def sample_df(spark):
    """
    Load a small sample of real robot data from the project's data/ folder.
    It loads UR3RTDE.csv (if found), otherwise UR5RTDE.csv.
    """

    base_path = os.path.abspath(os.path.join(os.getcwd(), "data"))

    possible_files = ["UR3RTDE.csv", "UR5RTDE.csv"]
    file_path = None

    for f in possible_files:
        path = os.path.join(base_path, f)
        if os.path.exists(path):
            file_path = path
            break

    if file_path is None:
        pytest.skip(f"No robot CSV found in data/ directory. Looked under: {base_path}")

    print(f"Using robot data file: {file_path}")

    # Read only a small sample
    df = spark.read.csv(file_path, header=True, inferSchema=True, sep=" ")
    df_sample = df.limit(5)
    return df_sample
