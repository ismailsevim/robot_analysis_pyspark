# Robotic Arm Analysis with PySpark

This repository showcases an integrated example combining PySpark, API communication, and LLM (Google Gemini) capabilities. It conducts statistical analyses on Universal Robots (UR3/UR5) joint telemetry data to detect potential systematic errors caused by temperature increases in the robotic arms, and leverages Gemini to evaluate the necessity of further modeling. The project is designed as a demonstration of applied techniques, rather than a general-purpose analysis or automation tool.

---

## Project Structure

```
robot_analysis_pyspark/
├── main.py                  # Entry point for running full analysis
├── .gitignore
│
├── data/                    # Robot CSV data (UR3RTDE.csv, UR5RTDE.csv)
│   └── .gitkeep             # Keeps the folder versioned without data
│
├── src/
│   ├── __init__.py
│   ├── data_checks.py       # Missing values, outliers, duplicates
│   ├── analysis.py          # Joint deviation and correlation analysis
│   └── gemini_client.py     # Connects to Gemini API for AI analysis
│
└── tests/
    ├── conftest.py          # PySpark session fixture
    ├── test_data_checks.py  # Unit tests for data validation
    ├── test_analysis.py     # Tests for statistical analysis
    └── test_gemini_client.py # Tests for Gemini integration
```

## Setup

1. Clone the repository
```
git clone https://github.com/ismailsevim/robot_analysis_pyspark.git
cd robot_analysis_pyspark
```
2. Install dependencies
```
pip install pyspark pytest google-generativeai
```
3. Add your robot data

The `data` folder is intentionally left empty due to file size limitations. Please download the datasets **UR3RTDE.csv** and **UR5RTDE.csv** from [this link](https://catalog.data.gov/dataset/process-and-robot-data-from-a-two-robot-workcell-representative-performing-representative-) and place them in the `data` folder before running the code.

4. Run tests
```
pytest -v
```

## Run the Analysis
```
python main.py --robot_name UR3 --api_key <YOUR_GEMINI_API_KEY>
```
This will:

  - Load the corresponding CSV file (e.g., data/UR3RTDE.csv)
  - Perform data validation and statistical checks
  - Summarize deviations and correlations
  - Send results to Gemini for AI-based interpretation

