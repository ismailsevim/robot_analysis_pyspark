from pyspark.sql import DataFrame, functions as F

def check_missing_values(df: DataFrame) -> DataFrame:
    """
    Check and count missing (null or NaN) values for each column.
    Returns a DataFrame with the count of missing entries.
    """
    print("\nChecking for missing (null/NaN) values...")

    missing_counts = df.select([
        F.count(F.when(F.col(c).isNull() | F.isnan(F.col(c)), c)).alias(c)
        for c in df.columns
    ])
    missing_counts.show(truncate=False)
    return missing_counts


def check_empty_strings(df: DataFrame) -> DataFrame:
    """
    Check and count empty or placeholder string values (e.g., '', 'N/A') for string columns.
    Returns a DataFrame summarizing empty string counts.
    """
    print("\nChecking for empty or 'N/A' string values...")

    string_cols = [c for c, t in df.dtypes if t == "string"]
    if not string_cols:
        print("No string columns found.")
        return df

    empty_counts = df.select([
        F.count(F.when((F.col(c) == "") | (F.lower(F.col(c)) == "n/a"), c)).alias(c)
        for c in string_cols
    ])
    empty_counts.show(truncate=False)
    return empty_counts

def check_numeric_ranges(df: DataFrame) -> None:
    """
    Summarize numeric columns to detect outliers or unrealistic values.
    Prints min, max, mean, stddev, and outlier count for each numeric column.
    If statistics cannot be calculated, replaces None with float('nan').
    """
    print("\nChecking numeric ranges, statistics, and outliers...")

    numeric_cols = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float")]
    if not numeric_cols:
        print("No numeric columns found.")
        return

    for col in numeric_cols:
        try:
            stats = df.select(
                F.min(col).alias("min"),
                F.max(col).alias("max"),
                F.mean(col).alias("mean"),
                F.stddev(col).alias("stddev"),
                F.expr(f'percentile_approx({col}, 0.25)').alias("q1"),
                F.expr(f'percentile_approx({col}, 0.75)').alias("q3")
            ).collect()[0]

            # Replace None values with float('nan')
            min_val = stats["min"] if stats["min"] is not None else float('nan')
            max_val = stats["max"] if stats["max"] is not None else float('nan')
            mean_val = stats["mean"] if stats["mean"] is not None else float('nan')
            stddev_val = stats["stddev"] if stats["stddev"] is not None else float('nan')
            q1 = stats["q1"] if stats["q1"] is not None else float('nan')
            q3 = stats["q3"] if stats["q3"] is not None else float('nan')

            # Calculate IQR bounds only if q1 and q3 are valid numbers
            if not (q1 != q1 or q3 != q3):  # check for NaN
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df.filter((F.col(col) < lower_bound) | (F.col(col) > upper_bound)).count()
            else:
                outliers = float('nan')

            print(f"• {col:30s} | min={min_val}, max={max_val}, "
                  f"mean={mean_val}, stddev={stddev_val}, "
                  f"outliers={outliers}")

        except Exception as e:
            print(f"• {col:30s} | Error calculating stats: {e}")



def check_duplicates(df: DataFrame) -> int:
    """
    Check for duplicate rows in the DataFrame.
    Returns the count of duplicated rows.
    """
    print("\nChecking for duplicate rows...")

    total_rows = df.count()
    unique_rows = df.dropDuplicates().count()
    duplicate_count = total_rows - unique_rows

    print(f"Total rows: {total_rows}, Unique rows: {unique_rows}, "
          f"Duplicate rows: {duplicate_count}")
    return duplicate_count


def check_distinct_values(df: DataFrame) -> list[str]:
    """
    Display distinct values for all integer columns and
    return a list of columns that have only one unique value.
    
    These columns are likely constants and can be safely removed
    from further analysis or modeling.
    """
    print("\nChecking distinct values for all integer columns...")

    # Identify integer columns
    int_cols = [c for c, t in df.dtypes if t in ("int", "integer", "bigint")]

    if not int_cols:
        print("No integer columns found in the DataFrame.")
        return []

    constant_cols = []

    for col in int_cols:
        distinct_vals = df.select(col).distinct()
        count = distinct_vals.count()

        print(f"\nColumn: {col} → {count} distinct values")
        distinct_vals.show(truncate=False)

        if count == 1:
            constant_cols.append(col)

    if constant_cols:
        print(f"\nColumns with a single unique value (constant): {constant_cols}")
    else:
        print("\nNo constant integer columns found.")

    return constant_cols
