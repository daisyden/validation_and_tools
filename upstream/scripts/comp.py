import os
import argparse
import pandas as pd

def compare():
    # Read all 4 CSV files
    df1 = pd.read_csv('/home/gta/daisyden/merge/periodic.csv', delimiter='\t')
    df2 = pd.read_csv('periodic.csv')
    
    # Rename columns to avoid conflicts
    df1 = df1.rename(columns={df1.columns[0]: 'file', df1.columns[1]: 'old'})
    df2 = df2.rename(columns={df2.columns[0]: 'file', df2.columns[1]: 'new'})
    
    # Merge sequentially
    merged_df = pd.merge(df1, df2, on='file', how='outer')
    
    print("Merge completed!")
    print(f"Total unique files: {len(merged_df)}")
    merged_df.to_csv('compare.csv', index=False)


if __name__ == "__main__":
    compare()
 
