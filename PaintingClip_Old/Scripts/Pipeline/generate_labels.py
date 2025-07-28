import argparse
import pandas as pd
import openpyxl
from typing import List

def simple_tokenize(text: str) -> List[str]:
    return text.strip().split()

def truncate_to_max_tokens(tokens: List[str], max_tokens: int) -> List[str]:
    return tokens[:max_tokens]

def generate_clip_label(row, max_tokens=77):
    parts = []
  
    title_val   = row.get('Title', None)
    creator_val = row.get('Creator', None)
    year_val    = row.get('Year', None)
    
    title   = str(title_val).strip() if pd.notna(title_val) else ""
    creator = str(creator_val).strip() if pd.notna(creator_val) else ""
    year    = str(int(year_val)) if pd.notna(year_val) else ""
    
    if title and year and creator:
        parts.append(f"{title} ({year}) by {creator}.")
    elif title and creator:
        parts.append(f"{title} by {creator}.")
    elif title:
        parts.append(f"{title}.")

    movements_val = row.get('Movements', None)
    movements     = str(movements_val).strip() if pd.notna(movements_val) else ""
    if movements and movements not in {'-', '-'}:
        parts.append(f"Style: {movements}.")
    
    depicts_val = row.get('Depicts', None)
    depicts     = str(depicts_val).strip() if pd.notna(depicts_val) else ""
    if depicts and depicts not in {'-', '-'}:
        items = [itm.strip() for itm in depicts.split(',') if itm.strip()]
        if items:
            parts.append("Depicts: " + ", ".join(items[:3]) + ".")
 
    sentence_val = row.get('TextualLabel', None)
    sentence     = str(sentence_val).strip() if pd.notna(sentence_val) else ""
    if sentence and sentence not in {'-', '-'}:
        parts.append(sentence)
    
    # Combine and truncate
    full_text = " ".join(parts)
    tokens    = simple_tokenize(full_text)
    tokens    = truncate_to_max_tokens(tokens, max_tokens)
    return " ".join(tokens).strip()

def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP-compatible labels for paintings, optionally for a subset of rows."
    )
    parser.add_argument(
        "--input", "-i",
        default="paintings_data_complete.xlsx",
        help="Path to the source Excel file (default: paintings_data.xlsx)"
    )
    parser.add_argument(
        "--output", "-o",
        default="paintings_token_labels.xlsx",
        help="Path for the output Excel file (default: paintings_data_labels.xlsx)"
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=None,
        help="Starting row index (0-based) to process; if omitted, starts at 0"
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="Ending row index (inclusive, 0-based); if omitted, processes to the last row"
    )
    args = parser.parse_args()

    # Load the full dataset
    df = pd.read_excel(args.input, engine="openpyxl")
    
    # Determine subset of rows
    start_idx = args.start or 0
    if args.end is not None:
        end_idx = args.end + 1
        subset = df.iloc[start_idx:end_idx].copy()
    else:
        subset = df.iloc[start_idx:].copy()
    
    # Generate labels
    subset['Textual Label'] = subset.apply(lambda r: generate_clip_label(r), axis=1)
    
    # Prepare output
    output_df = subset[['Title', 'File Name', 'Textual Label']].copy()
    
    # Write to Excel and adjust column widths
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        output_df.to_excel(writer, index=False, sheet_name="Labels")
        ws = writer.sheets['Labels']
        
        # Title column: wide enough for example long title or actual max
        example_title = "Where Do We Come From? What Are We? Where Are We Going?"
        min_title_w   = len(example_title) + 2
        actual_title_w= output_df['Title'].astype(str).map(len).max() + 2
        ws.column_dimensions['A'].width = max(min_title_w, actual_title_w)
        
        # File Name column: based on actual data
        fn_width = output_df['File Name'].astype(str).map(len).max() + 2
        ws.column_dimensions['B'].width = fn_width
        
        # Textual Label column: based on actual data
        lbl_width = output_df['Textual Label'].astype(str).map(len).max() + 2
        ws.column_dimensions['C'].width = lbl_width
    
    total = len(output_df)
    print(f"Processed rows {start_idx}"
          f"{'-' + str(args.end) if args.end is not None else ''}: "
          f"generated {total} labels, saved to '{args.output}'.")

if __name__ == "__main__":
    main()
