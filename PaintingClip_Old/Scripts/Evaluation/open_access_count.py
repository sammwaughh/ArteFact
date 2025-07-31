import os
from pathlib import Path
import zipfile

import pandas as pd
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent  # Project/
DATA_DIR = BASE_DIR / "Dataset" / "Archive" / "Excel-Files"
OUTPUT_DIR = BASE_DIR / "Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize counters
total_all = 0
total_accessible = 0
total_downloaded = 0
processed_files = 0
skipped_files = 0

# Iterate over Excel files
for excel_file in DATA_DIR.iterdir():
    if excel_file.name.startswith("~") or excel_file.suffix.lower() not in (
        ".xlsx",
        ".xls",
    ):
        skipped_files += 1
        continue
    try:
        engine = "openpyxl" if excel_file.suffix.lower() == ".xlsx" else "xlrd"
        df_all = pd.read_excel(excel_file, sheet_name="All", engine=engine)
        df_acc = pd.read_excel(excel_file, sheet_name="Accessible", engine=engine)
        df_dl = pd.read_excel(excel_file, sheet_name="Downloaded", engine=engine)
    except (zipfile.BadZipFile, ValueError, FileNotFoundError) as e:
        print(f"Skipping {excel_file.name}: {e}")
        skipped_files += 1
        continue

    total_all += len(df_all)
    total_accessible += len(df_acc)
    total_downloaded += len(df_dl)
    processed_files += 1

# Build summary table with percentages only
percentages = [
    100.0,
    round(total_accessible / total_all * 100, 1),
    round(total_downloaded / total_all * 100, 1),
]
summary_df = pd.DataFrame(
    {
        "Category": ["All", "Accessible", "Downloaded"],
        "Percentage": [f"{p}%" for p in percentages],
    }
)

# Plot and save as PNG
fig, ax = plt.subplots(figsize=(3, 1.5))
ax.axis("off")
tbl = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc="center",
    loc="center",
)

# Style header row and first column
light_grey = "#f0f0f0"
for (row, col), cell in tbl.get_celld().items():
    if row == 0 or col == 0:
        cell.set_facecolor(light_grey)
        cell.set_text_props(weight="bold")

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)

output_path = OUTPUT_DIR / "paper_percentages_summary.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved percentage table to {output_path}")
print(f"Processed files: {processed_files}")
print(f"Skipped files: {skipped_files}")
