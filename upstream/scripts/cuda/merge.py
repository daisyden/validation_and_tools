import os
import argparse
import pandas as pd

def merge(f1: str, f2: str, f3: str, f4: str, output: str):
    backend1 = os.path.basename(f1).split('.')[0]
    backend2 = os.path.basename(f2).split('.')[0]
    backend3 = os.path.basename(f3).split('.')[0]

    # Read all 4 CSV files
    df1 = pd.read_csv(f1, delimiter='|', engine='python')
    df2 = pd.read_csv(f2, delimiter='|', engine='python')
    df3 = pd.read_csv(f3, delimiter='|', engine='python')
    df4 = pd.read_csv(f4)

    if len(df1) > 0 and df1.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df1 = df1.iloc[:-1]
    if len(df2) > 0 and df2.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df2 = df2.iloc[:-1]
    if len(df3) > 0 and df3.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df3 = df3.iloc[:-1]
    if len(df4) > 0 and df4.iloc[-1, 1] == " **Total** " :  # iloc[-1, 1] gets last row, 2nd column
        df4 = df4.iloc[:-1]
 

    df1 = df1[["UT", "Test cases", "Passed", "Skipped", "Failures"]]
    df2 = df2[["UT", "Test cases", "Passed", "Skipped", "Failures"]]
    df3 = df3[["UT", "Test cases", "Passed", "Skipped", "Failures"]]
    df4 = df4[[" Testfile ", " Test cases ", " Passed ", " Skipped ", " Failures "]]
    
    # Rename columns to avoid conflicts
    df1 = df1.rename(columns={df1.columns[0]: 'Testfile', df1.columns[1]: f"{backend1}-total", df1.columns[2]: f"{backend1}-PASSED", df1.columns[3]: f"{backend1}-Skipped", df1.columns[4]: f"{backend1}-Failed"})
    df2 = df2.rename(columns={df2.columns[0]: 'Testfile', df2.columns[1]: f"{backend2}-total", df2.columns[2]: f"{backend2}-PASSED", df2.columns[3]: f"{backend2}-Skipped", df2.columns[4]: f"{backend2}-Failed"})
    df3 = df3.rename(columns={df3.columns[0]: 'Testfile', df3.columns[1]: f"{backend3}-total", df3.columns[2]: f"{backend3}-PASSED", df3.columns[3]: f"{backend3}-Skipped", df3.columns[4]: f"{backend3}-Failed"})
    df4 = df4.rename(columns={df4.columns[0]: 'Testfile', df4.columns[1]: f"{backend3}-total", df4.columns[2]: f"{backend3}-PASSED", df4.columns[3]: f"{backend3}-Skipped", df4.columns[4]: f"{backend3}-Failed"})

    numeric_cols = df1.columns[1:]
    df1 = df1.groupby(df1.columns[0])[numeric_cols].sum().reset_index()

    numeric_cols = df2.columns[1:]
    df2 = df2.groupby(df2.columns[0])[numeric_cols].sum().reset_index()

    # Merge sequentially
    merged_df = pd.merge(df1, df2, on='Testfile', how='outer')
    
    print("Merge completed!")
    print(f"Total unique files: {len(merged_df)}")
    merged_df.to_csv('upstream.csv', index=False)

    df_local = pd.concat([df3, df4], axis=0, ignore_index=True)
    df_local = df_local.rename(columns={df_local.columns[0]: 'Testfile', df_local.columns[1]: f"{backend3}-total", df_local.columns[2]: f"{backend3}-PASSED", df_local.columns[3]: f"{backend3}-Skipped", df_local.columns[4]: f"{backend3}-Failed" })

    df_local["Testfile"] = df_local["Testfile"].astype(str)
    merged_df["Testfile"] = merged_df["Testfile"].astype(str)
    
    merged_df = pd.merge(merged_df, df_local, on='Testfile', how='outer')

    df_c = pd.read_csv("data/collected.csv") 

    df_c = df_c.rename(columns={df_c.columns[0]: 'Testfile', df_c.columns[1]: f"total", df_c.columns[2]: f"deselected", df_c.columns[3]: f"selected" })
    merged_df = pd.merge(merged_df, df_c, on='Testfile', how='outer')

    # Fill NaN values with 0 and convert to integers
    for col in merged_df.columns:
        if col != "Testfile":
            merged_df[col] = merged_df[col].fillna(0).astype(int)
 
    merged_df.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge two .csv files of different workflow')
    parser.add_argument('--f1', default='data/summary/cuda.csv', help='The first summary is for reference backend, cuda backend by default, default is data/summary/cuda.csv')
    parser.add_argument('--f2', default='data/summary/stock_xpu.csv', help='The 2nd summary is for stock pytorch xpu backend by default, default is data/summary/stock_xpu.csv')
    parser.add_argument('--f3', default='data/summary/xpu-ops.csv', help='The 3rd summary is for xpu ops, default is data/summary/xpu-ops.csv')
    parser.add_argument('--f4', default='data/summary/dist_summary_parsed.csv', help='The 4th summary is for xpu distributed, default is data/summary/dist_summary_parsed.csv')
    parser.add_argument('-o', '--output', default='merged.csv', help='The output csv file (default: merged.csv)')

    args = parser.parse_args()

    merge(args.f1, args.f2, args.f3, args.f4, args.output)
 
