from pyspark.sql import SparkSession
from src.data_checks import (
    check_missing_values, check_empty_strings,
    check_numeric_ranges, check_duplicates, check_distinct_values)
from src.analysis import analyze_joint_differences_spark
from src.gemini_client import gemini
import argparse

def main():
    
    # --- Parse input arguments ---
    parser = argparse.ArgumentParser(description="UR Robot Data Analysis with Gemini AI")
    parser.add_argument("--robot_name", type=str, required=True,
                        help="Name of the robot (e.g., UR3, UR5)")
    parser.add_argument("--api_key", type=str, required=True,
                        help="Google Gemini API key for analysis")
    args = parser.parse_args()

    robot_name = args.robot_name
    api_key = args.api_key

    # --- Step 0: Initialize Spark ---
    spark = SparkSession.builder.appName(f"{robot_name}RobotAnalysis").getOrCreate()
    file_path = f"data/{robot_name}RTDE.csv"

    print(f"\nLoading data for {robot_name} from: {file_path}")
    df = spark.read.csv(file_path, header=True, inferSchema=True, sep=" ")

    print("\nData successfully loaded.")
    n_rows = df.count()
    print(f"Number of rows: {n_rows}")
    n_cols = len(df.columns)
    print(f"Number of columns: {n_cols}")
    df.printSchema()
    df.show(5)

    # --- Step 1: Validation ---
    check_missing_values(df)
    check_empty_strings(df)
    #check_numeric_ranges(df)
    check_duplicates(df)
    constant_int_cols = check_distinct_values(df)

    # --- Step 2: Cleaning ---
    if constant_int_cols:
        print(f"\nRemoving constant columns: {constant_int_cols}")
        df = df.drop(*constant_int_cols)

    # --- Step 3: Statistical Analysis & Prompt ---
    summary_list, prompt = analyze_joint_differences_spark(df, robot_name)

    # --- Step 4: Gemini Analysis ---
    print("\nSending analysis to Gemini...")
    gemini(summary_list, prompt, api_key=api_key)


if __name__ == "__main__":
    main()