import os
import argparse
import pandas as pd

def merge(f1: str, f2: str, output: str):
    backend1 = os.path.basename(f1).split('.')[0]
    backend2 = os.path.basename(f2).split('.')[0]

    # Read all 4 CSV files
    df1 = pd.read_csv(f1, delimiter='|', engine='python')
    df2 = pd.read_csv(f2, delimiter='|', engine='python')

    if len(df1) > 0 and df1.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df1 = df1.iloc[:-1]
    if len(df2) > 0 and df2.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df2 = df2.iloc[:-1]

    df1 = df1[["UT", "Test cases", "Passed", "Skipped", "Failures"]]
    df2 = df2[["UT", "Test cases", "Passed", "Skipped", "Failures"]]
    
    # Rename columns to avoid conflicts
    df1 = df1.rename(columns={df1.columns[0]: 'Testfile', df1.columns[1]: f"{backend1}-total", df1.columns[2]: f"{backend1}-PASSED", df1.columns[3]: f"{backend1}-Skipped", df1.columns[4]: f"{backend1}-Failed"})
    df2 = df2.rename(columns={df2.columns[0]: 'Testfile', df2.columns[1]: f"{backend2}-total", df2.columns[2]: f"{backend2}-PASSED", df2.columns[3]: f"{backend2}-Skipped", df2.columns[4]: f"{backend2}-Failed"})

    numeric_cols = df1.columns[1:]
    df1 = df1.groupby(df1.columns[0])[numeric_cols].sum().reset_index()

    numeric_cols = df2.columns[1:]
    df2 = df2.groupby(df2.columns[0])[numeric_cols].sum().reset_index()

    # Merge sequentially
    merged_df = pd.merge(df1, df2, on='Testfile', how='outer')

    substring = '/inductor/'
    merged_df = merged_df[merged_df.iloc[:, 0].str.contains(substring, na=False)]
    
    print("Merge completed!")
    print(f"Total unique files: {len(merged_df)}")
    merged_df.to_csv('merged_inductor.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge two .csv files of different workflow')
    parser.add_argument('--f1', default='data/summary/cuda.csv', help='The reference summary file, default is data/summary/cuda.csv')
    parser.add_argument('--f2', default='data/summary/stock_xpu.csv', help='The 2nd summary is for stock pytorch xpu backend by default, default is data/summary/stock_xpu.csv')
    parser.add_argument('-o', '--output', default='merged_inductor.csv', help='The output csv file (default: merged_inductor.csv)')

    args = parser.parse_args()

    merge(args.f1, args.f2, args.output)
 
