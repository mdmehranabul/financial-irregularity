# financial-irregularity
To detect resolved and unresolved cases. Also, summarize and suggest next steps.

*   **Problem and Approach:** This project addresses the challenge of automating financial discrepancy categorization and resolution. The approach involves preprocessing financial data, categorizing discrepancies using keywords and potentially LLM semantic understanding, and automating actions based on resolution status. BART (`facebook/bart-large-cnn`) is employed as the LLM for text generation tasks (summarization, next steps, and pattern identification).
*   **Model Explanation:** BART is chosen for its strong performance in text generation and summarization tasks. The `facebook/bart-large-cnn` model is used for its balance between performance and resource requirements.  Preprocessing steps include handling missing values, converting data types, and normalizing text for consistent keyword matching. 
*   **Evaluation:** The solution is evaluated based on accuracy (how well it categorizes discrepancies), performance (processing speed), code quality, scalability, and explainability (clarity of LLM prompts and outputs).  A portion of the input data should be reserved as a test set and kept separate from the development process.

**Testing Setup:**

To test the solution:

1.  **Data:** Place sample input CSV files in the `data/` directory.  The files should have the expected structure and column names.
2.  **Execution:** Run the main script using the command `python final_code.py`.
3.  **Verification:** Check the `output/` directory for the processed output files.  Verify that the discrepancies are correctly categorized and that the generated summaries, next steps, and patterns are reasonable.


* *** Testing Setup ***
This solution has been tested through a combination of methods to ensure its functionality and robustness:

Unit Testing (Partial): While full-fledged unit tests aren't included in this initial version, the core logic of the preprocess_and_categorize function has been tested by verifying the shape and contents of the resulting DataFrames.  The keyword matching logic within process_resolution has been tested with various sample comments to confirm correct identification of resolved/unresolved cases.

Integration Testing: The integration of the different components (data loading, preprocessing, LLM interaction, file saving) has been tested by running the main script with sample input CSV files (recon_data_raw.csv and recon_data_reply.csv). The output files (matched.csv, not_found_sys_b.csv, not_matched.csv, resolved_cases.csv, and unresolved_cases.csv) have been examined to ensure that the data is processed correctly and the LLM generates reasonable summaries and next steps.

Edge Case Testing: The code has been tested with a few edge cases:

Missing Comments: The process_resolution function now handles cases where the "Comments" field is missing or empty by using .get('Comments', '') and .strip().
Varying Comment Formats: The regular expression for keyword matching (\bresolved\b, \bunresolved\b) handles variations in capitalization and ensures that only whole words are matched.
Numeric Data Handling: The preprocess_and_categorize function fills missing numeric values with 0 to avoid errors during calculations.


* *** How to Test the Solution:***

To test the solution on your side, follow these steps:

Environment Setup: Set up the Python environment as described in the "Integration" section of the README. This involves cloning the repository, creating a virtual environment, and installing the required dependencies using requirements.txt.

Input Data: Prepare your input CSV files (recon_data_raw.csv and recon_data_reply.csv) and place them in the data/ directory. Ensure that the files have the required columns (txn_ref_id or order_id, sys_a_amount_attribute_1, sys_a_date, Comments, Transaction ID). You can use the provided sample data files as a starting point, but it's crucial to test with your own data to verify that the code works as expected.

Execution: Run the script using the command python final_code.py. 

Output Verification: Check the output/ directory for the generated CSV files.

Verify that matched.csv, not_found_sys_b.csv, and not_matched.csv contain the correct data based on the recon_sub_status field.
Examine resolved_cases.csv and unresolved_cases.csv to confirm that the resolved and unresolved cases are correctly separated.
For the unresolved cases in unresolved_cases.csv, check that the "summary" and "next_steps" columns contain reasonable text generated by the LLM.
Edge Case Testing: It's highly recommended to test with various edge cases, such as:

Empty or missing "Comments" fields.
Comments with different phrasing related to resolution (e.g., "Issue is fixed," "Problem has been resolved").
Comments with mixed case and extra whitespace.
Large datasets to evaluate the performance and scalability.
Invalid or missing numeric or date data.
Accuracy Evaluation : If you have labeled data, you can compare the solution's output with the true labels to calculate accuracy metrics (precision, recall, F1-score).