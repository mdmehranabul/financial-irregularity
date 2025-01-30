
import pandas as pd
import os
import json
from transformers import pipeline
import re

# Directory information
DATA_DIR = "data"  # Directory containing input CSV files
OUTPUT_DIR = "output" # Directory to store output CSV files
RESOLVED_DIR = os.path.join(OUTPUT_DIR, "resolved")
UNRESOLVED_DIR = os.path.join(OUTPUT_DIR, "unresolved")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESOLVED_DIR, exist_ok=True)
os.makedirs(UNRESOLVED_DIR, exist_ok=True)

def preprocess_and_categorize(input_file):
    df = pd.read_csv(input_file).head(200)

    # Handle missing/invalid data
    df['sys_a_amount_attribute_1'].fillna(0, inplace=True)
    df['sys_b_amount_attribute_1'].fillna(0, inplace=True)
    df['sys_a_date'] = pd.to_datetime(df['sys_a_date'], errors='coerce')
    df['sys_b_date'] = pd.to_datetime(df['sys_b_date'], errors='coerce')
    
    # Convert recon_sub_status to dictionary and extract categories
    def extract_status(status, category):
        try:
            status_dict = json.loads(status.replace("'", '"'))  # Convert single to double quotes if needed
            return category in status_dict.values()
        except (json.JSONDecodeError, TypeError):
            return False  # If parsing fails, assume it's not a match

    df['is_matched'] = df['recon_sub_status'].apply(lambda x: extract_status(x, "Matched"))
    df['is_not_found_sys_b'] = df['recon_sub_status'].apply(lambda x: extract_status(x, "Not Found-SysB"))
    df['is_not_matched'] = df['recon_sub_status'].apply(lambda x: extract_status(x, "Not Matched"))

    # Filter the data into separate categories
    matched = df[df['is_matched']][['txn_ref_id', 'sys_a_amount_attribute_1', 'sys_a_date']].copy()
    not_found_sys_b = df[df['is_not_found_sys_b']][['txn_ref_id', 'sys_a_amount_attribute_1', 'sys_a_date']].copy()
    not_matched = df[df['is_not_matched']][['txn_ref_id', 'sys_a_amount_attribute_1', 'sys_a_date']].copy()
    
    # Format date column
    for category_df in [matched, not_found_sys_b, not_matched]:
        category_df['date'] = category_df['sys_a_date'].dt.strftime('%Y-%m-%d')
        category_df.drop(columns=['sys_a_date'], inplace=True)
        category_df.rename(columns={'txn_ref_id': 'order_id'}, inplace=True) #Rename Here
    
    return matched, not_found_sys_b, not_matched

def upload_file(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"File uploaded to: {output_path}")

llm = pipeline('text2text-generation', model='facebook/bart-large-cnn')

def process_resolution(resolution_file, not_found_data):
    resolution_df = pd.read_csv(resolution_file, encoding="ISO-8859-1").head(200)
    merged_df = pd.merge(not_found_data, resolution_df, left_on='order_id', right_on='Transaction ID', how='left')
    
    # Lists to store resolved and unresolved rows
    resolved_cases = []
    unresolved_cases = []
    unresolved_details = []  # New list to store summary & next steps

    for index, row in merged_df.iterrows():
        order_id = row['order_id']
        amount = row['amount']
        date = row['date']
        comment = str(row.get('Comments', '')).strip()  # Ensure comment is a string

        if re.search(r'\bresolved\b', comment, re.IGNORECASE):  
            resolved_cases.append([order_id, amount, date, comment])
        else:
            summary = generate_summary(comment, llm)  
            next_steps = suggest_next_steps(comment, llm)  
            unresolved_cases.append([order_id, amount, date, comment])
            unresolved_details.append([order_id, comment, summary, next_steps])  # Store in new file

    # Convert lists to DataFrames
    resolved_df = pd.DataFrame(resolved_cases, columns=['order_id', 'amount', 'date', 'comment'])
    unresolved_df = pd.DataFrame(unresolved_cases, columns=['order_id', 'amount', 'date', 'comment'])
    unresolved_details_df = pd.DataFrame(unresolved_details, columns=['order_id', 'comment', 'summary', 'next_steps'])  # ✅ New DataFrame

    # Save consolidated files
    resolved_filepath = os.path.join(RESOLVED_DIR, "resolved_cases.csv")
    unresolved_filepath = os.path.join(UNRESOLVED_DIR, "unresolved_cases.csv")
    unresolved_details_filepath = os.path.join(UNRESOLVED_DIR, "unresolved_cases_details.csv")  # New file

    resolved_df.to_csv(resolved_filepath, index=False)
    unresolved_df.to_csv(unresolved_filepath, index=False)
    unresolved_details_df.to_csv(unresolved_details_filepath, index=False)  #Save new file

    print(f"Consolidated Resolved cases saved at: {resolved_filepath}")
    print(f"Consolidated Unresolved cases saved at: {unresolved_filepath}")
    print(f"Unresolved Cases Details saved at: {unresolved_details_filepath}")  # Print new file path


def identify_pattern(comment, llm):  # Add llm as an argument
    prompt = f"Identify the key pattern or reason for resolution in this financial discrepancy comment: {comment}"
    pattern = llm(prompt)[0]['generated_text']  # Extract from BART's output
    return pattern.strip()

def generate_summary(comment, llm):  # Add llm as an argument
    prompt = f"Summarize why this financial discrepancy is unresolved: {comment}"
    summary = llm(prompt)[0]['generated_text']  # Extract from BART's output
    return summary.strip()

def suggest_next_steps(comment, llm):  # Add llm as an argument
    prompt = f"Suggest next steps to resolve this financial discrepancy: {comment}"
    next_steps = llm(prompt)[0]['generated_text']  # Extract from BART's output
    return next_steps.strip()

def send_email(order_id, status, *args):  # *args for variable number of arguments
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = f"Financial Discrepancy Update - Order ID: {order_id}"

    body = f"Order ID: {order_id}\nStatus: {status}\n"
    for arg in args:  # Add summary, next steps, or pattern to email body
        body += f"{arg}\n"

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Example: Gmail
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, EMAIL_RECIPIENT, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Email sending failed: {e}")

if __name__ == "__main__":
    DATA_DIR = "./data"
    OUTPUT_DIR = "./output"
    
    raw_data_file = os.path.join(DATA_DIR, "recon_data_raw.csv")
    resolution_file = os.path.join(DATA_DIR, "recon_data_reply.csv")

    matched_data, not_found_data, not_matched_data = preprocess_and_categorize(raw_data_file)
    
    matched_file_path = os.path.join(OUTPUT_DIR, "matched.csv")
    not_found_file_path = os.path.join(OUTPUT_DIR, "not_found_sys_b.csv")
    not_matched_file_path = os.path.join(OUTPUT_DIR, "not_matched.csv")
    
    upload_file(matched_data, matched_file_path)
    upload_file(not_found_data, not_found_file_path)
    upload_file(not_matched_data, not_matched_file_path)
    
    process_resolution(resolution_file, not_found_data)




