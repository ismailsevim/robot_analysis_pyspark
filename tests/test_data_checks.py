from src.data_checks import (
    check_missing_values, check_empty_strings,
    check_numeric_ranges, check_duplicates, check_distinct_values
)

def test_check_missing_values(sample_df):
    result = check_missing_values(sample_df)
    assert result is not None

def test_check_empty_strings(sample_df):
    result = check_empty_strings(sample_df)
    assert result is not None

def test_check_numeric_ranges(sample_df):
    check_numeric_ranges(sample_df)  # Just ensure it runs without error

def test_check_duplicates(sample_df):
    duplicates = check_duplicates(sample_df)
    assert isinstance(duplicates, int)

def test_check_distinct_values(sample_df):
    constants = check_distinct_values(sample_df)
    assert isinstance(constants, list)
