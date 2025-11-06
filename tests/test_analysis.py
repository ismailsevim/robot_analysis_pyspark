from src.analysis import analyze_joint_differences_spark

def test_analyze_joint_differences_spark(sample_df):
    summary_list, prompt = analyze_joint_differences_spark(sample_df, robot_name="UR3")
    
    # Basic checks
    assert isinstance(summary_list, list)
    assert all("joint" in s and "mean_diff" in s and "std_diff" in s and "correlation" in s for s in summary_list)
    assert isinstance(prompt, str)
    assert "Robot: UR3" in prompt
