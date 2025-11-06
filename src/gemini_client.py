from google import genai

def gemini(summary_list, prompt, api_key="", model="gemini-2.5-flash"):
    """
    Send robot joint deviation summary and prompt to Gemini for analysis.

    Parameters
    ----------
    summary_list : list of dicts
        Output from analyze_joint_differences_spark()
    prompt : str
        Natural-language prompt describing what to analyze
    api_key : str
        Your Gemini API key
    model : str
        Gemini model name (default = "gemini-2.5-flash")

    Returns
    -------
    ai_text : str
        The model’s analytical response (or an error message)
    """
    if not prompt or len(prompt.strip()) == 0:
        print("Empty prompt provided.")
        return None

    # Create Gemini client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print("Failed to initialize Gemini client:", e)
        return None

    # Build structured message
    analysis_prompt = (
        "Robot Joint Analysis Request\n"
        "===========================\n\n"
        "Instructions:\n"
        "1. Examine the deviation between `target_q` and `actual_q` for each joint.\n"
        "2. Identify significant deviations or systematic errors.\n"
        "3. Check correlation of deviations with target positions.\n"
        "4. Suggest real-time compensation strategies where necessary.\n\n"
        "Joint Summary Data:\n"
        + "\n".join([
            f"- Joint {s['joint']}: mean_diff = {s['mean_diff']:.6e} rad, "
            f"std_diff = {s['std_diff']:.6e} rad, correlation = {s['correlation']:.6f}"
            for s in summary_list
        ])
        + "\n\nPlease provide a clear, concise analysis highlighting which joints (if any) "
          "show significant deviations or systematic behavior, and propose practical real-time compensation methods."
    )

    # Send request to Gemini
    try:
        response = client.models.generate_content(
            model=model,
            contents=analysis_prompt
        )
        ai_text = response.text
        print("\nGemini Analysis:\n")
        print(ai_text)
        return ai_text
    except Exception as e:
        print("Gemini API request failed:", e)
        return None
