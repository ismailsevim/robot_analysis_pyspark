from pyspark.sql import functions as F
import math

def analyze_joint_differences_spark(
    df,
    robot_name="Robot",
    real_prefix="actual_q_",
    desired_prefix="target_q_",
    corr_threshold=0.9,
    build_prompt=True
):
    """
    Analyze joint position differences and correlations for a PySpark DataFrame,
    and optionally generate a natural-language prompt for Gemini.
    
    This version includes robust error handling to set failed calculations
    (like correlation on constant columns) to float('nan').

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame containing columns like actual_q_0..5 and target_q_0..5
    robot_name : str
        Name of the robot (e.g., "UR3", "UR5")
    real_prefix : str
        Prefix for real joint values.
    desired_prefix : str
        Prefix for desired joint values.
    corr_threshold : float
        Threshold below which correlations are considered meaningful deviations.
    build_prompt : bool
        If True, generates a prompt string summarizing results for Gemini.

    Returns
    -------
    summary_list : list of dicts
        Each dict has {joint, mean_diff, std_diff, correlation}
    prompt : str or None
        Natural-language prompt for Gemini if build_prompt=True, else None
    """
    summary_list = []

    for j in range(6):
        real_col = f"{real_prefix}{j}"
        desired_col = f"{desired_prefix}{j}"

        if real_col not in df.columns or desired_col not in df.columns:
            print(f"Missing columns for joint {j}: {real_col}, {desired_col}")
            continue

        # Initialize stats with NaN in case of error or non-computable result
        stats = {
            "mean_diff": float('nan'),
            "std_diff": float('nan'),
            "correlation": float('nan')
        }

        try:
            # Compute mean, std, and correlation using PySpark functions
            diff_col = F.col(real_col) - F.col(desired_col)
            result = df.select(
                F.mean(diff_col).alias("mean_diff"),
                F.stddev(diff_col).alias("std_diff"),
                F.corr(real_col, desired_col).alias("correlation")
            ).collect()
            
            # If computation was successful, extract the row and handle potential None/NaN values
            if result and result[0]:
                row = result[0].asDict()
                
                # Check for None values (PySpark's way of indicating non-computable results, 
                # e.g., correlation of a constant series) and replace with NaN
                stats["mean_diff"] = row["mean_diff"] if row["mean_diff"] is not None else float('nan')
                stats["std_diff"] = row["std_diff"] if row["std_diff"] is not None else float('nan')
                stats["correlation"] = row["correlation"] if row["correlation"] is not None else float('nan')

            else:
                print(f"Warning: Spark computation returned empty result for joint {j}. Using NaN.")

        except Exception as e:
            # Catch PySpark execution errors (e.g., if a column contains non-numeric data)
            print(f"Error computing stats for joint {j}: {e}. Setting all values to NaN.")
            # stats remains the NaN initialized dictionary

        # Append computed (or NaN) results to the summary list
        summary_list.append({
            "joint": j,
            "mean_diff": stats["mean_diff"],
            "std_diff": stats["std_diff"],
            "correlation": stats["correlation"]
        })
        
        # Check correlation against threshold (safely checking if it's not NaN)
        current_corr = stats["correlation"]
        
        # Check if the value is a number (not NaN) and if its absolute value is below the threshold
        if not math.isnan(current_corr) and abs(current_corr) < corr_threshold:
            print(f"Joint {j}: correlation ({current_corr:.3f}) below threshold {corr_threshold}")

    # Build a clean, structured prompt for Gemini
    prompt = None
    if build_prompt:
        prompt_lines = [
            f"Robot: {robot_name}",
            "Objective: Analyze joint deviations and propose real-time compensation strategies.\n",
            "Instructions:",
            "1. Examine deviations between `target_q` and `actual_q` for each joint.",
            "2. Identify meaningful deviations or systematic errors.",
            "3. Evaluate correlation of deviations with target positions.",
            "4. Suggest practical real-time compensation methods.\n",
            "Joint Statistics:"
        ]

        for s in summary_list:
            # Python's f-strings handle float('nan') gracefully in formatting
            prompt_lines.append(
                f"- Joint {s['joint']}: mean deviation = {s['mean_diff']:.6e} rad, "
                f"std deviation = {s['std_diff']:.6e} rad, "
                f"correlation with target_q = {s['correlation']:.6f}"
            )

        prompt_lines.append(
            "\nPlease provide a structured analysis: highlight significant deviations, "
            "explain systematic errors if any, and propose practical compensation methods."
        )

        prompt = "\n".join(prompt_lines)

    return summary_list, prompt