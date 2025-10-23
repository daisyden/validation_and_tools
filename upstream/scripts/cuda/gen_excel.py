import pandas as pd
import os
from glob import glob

def csv_to_excel_with_sheets(csv_folder, output_file):
    """
    Merge all CSV files in a folder into an Excel file with multiple sheets
    Each CSV becomes a separate sheet named after the file
    """
    # Find all CSV files
    csv_files = glob(os.path.join(csv_folder, "*.csv"))
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for csv_file in csv_files:
            # Get filename without extension for sheet name
            sheet_name = os.path.splitext(os.path.basename(csv_file))[0]
            
            # Read CSV and write to Excel
            df = pd.read_csv(csv_file)
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Sheet names max 31 chars
    
    print(f"Created {output_file} with {len(csv_files)} sheets")

# Usage
csv_to_excel_with_sheets('./data/report', 'merged_data.xlsx')
