import os
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Specify the inclusive range (1-indexed) of rows to process from painters.xlsx.
start_row = 87
end_row = 100

# Read painters.xlsx to get the list of painter (folder) names from the "Query String" column.
df = pd.read_excel("painters.xlsx")
folder_names = df.loc[start_row - 1:end_row - 1, "Query String"].tolist()

# Define base directories.
input_base = "All Works"                  # Where the input PDF folders are located.
output_base = "Small Markdown"             # Where output directories will be created.
excel_base = "Works Metadata"            # Directory containing the {painter}_works.xlsx files.

def process_pdf(folder, pdf_file):
    """Process a single PDF file using marker_single with a one-hour timeout."""
    input_folder = os.path.join(input_base, folder)
    output_dir = os.path.join(output_base, folder)
    os.makedirs(output_dir, exist_ok=True)
    input_pdf = os.path.join(input_folder, pdf_file)
    cmd = f'marker_single "{input_pdf}" --output_format markdown --output_dir "{output_dir}"'
    print(f"Processing file: {input_pdf}")
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for file: {input_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing file: {input_pdf}. Error: {e}")
    print(f"Finished processing file: {input_pdf}\n")

# Collect tasks from each painter folder.
tasks = []
for folder in folder_names:
    input_folder = os.path.join(input_base, folder)
    excel_file = os.path.join(excel_base, f"{folder}_works.xlsx")
    try:
        works_df = pd.read_excel(excel_file, sheet_name="For Markdown")
    except Exception as e:
        print(f"Error reading sheet 'For Markdown' in {excel_file}: {e}")
        continue
    # Filter rows where "file size" (assumed to be in KB) is less than 10000.
    filtered_df = works_df[works_df["file size"] < 10000]
    # Get the top 10 PDF file names from the first column of the filtered DataFrame.
    pdf_files = filtered_df.iloc[50:1000, 0].tolist()
    for pdf_file in pdf_files:
        tasks.append((folder, pdf_file))

# Use ThreadPoolExecutor to process PDF files concurrently.
max_workers = 5  # Adjust as needed based on your system.
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(lambda args: process_pdf(*args), tasks)
