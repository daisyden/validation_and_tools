import os
import argparse
import pandas as pd

def merge(output: str):
    # Read all 4 CSV files
    df1 = pd.read_csv('data/rocm.csv', delimiter='|', engine='python')
    df2 = pd.read_csv('data/stock_xpu.csv', delimiter='|', engine='python')
    df3 = pd.read_csv('data/xpu-ops.csv', delimiter='|', engine='python')
    df4 = pd.read_csv('data/dist_summary_parsed.csv')
 
    import pdb
    pdb.set_trace()
    df1 = df1[["UT", "Test cases", "Passed", "Skipped", "Failures"]]
    df2 = df2[["UT", "Test cases", "Passed", "Skipped", "Failures"]]
    df3 = df3[["UT", "Test cases", "Passed", "Skipped", "Failures"]]
    df4 = df4[[" Testfile ", " Test cases ", " Passed ", " Skipped ", " Failures "]]
    
    # Rename columns to avoid conflicts
    df1 = df1.rename(columns={df1.columns[0]: 'Testfile', df1.columns[1]: 'rocm-total', df1.columns[2]: 'rocm-PASSED', df1.columns[3]: 'rocm-Skipped', df1.columns[4]: 'rocm-Failed'})
    df2 = df2.rename(columns={df2.columns[0]: 'Testfile', df2.columns[1]: 'xpu_upstream-total', df2.columns[2]: 'xpu_upstream-PASSED', df2.columns[3]: 'xpu_upstream-Skipped', df2.columns[4]: 'xpu_upstream-Failed'})
    df3 = df3.rename(columns={df3.columns[0]: 'Testfile', df3.columns[1]: 'xpu_ops-total', df3.columns[2]: 'xpu_ops-PASSED', df3.columns[3]: 'xpu_ops-Skipped', df3.columns[4]: 'xpu_ops-Failed'})
    df4 = df4.rename(columns={df4.columns[0]: 'Testfile', df4.columns[1]: 'xpu_ops-total', df4.columns[2]: 'xpu_ops-PASSED', df4.columns[3]: 'xpu_ops-Skipped', df4.columns[4]: 'xpu_ops-Failed'})

    numeric_cols = df1.columns[1:]
    df1 = df1.groupby(df1.columns[0])[numeric_cols].sum().reset_index()

    numeric_cols = df2.columns[1:]
    df2 = df2.groupby(df2.columns[0])[numeric_cols].sum().reset_index()

    # Merge sequentially
    merged_df = pd.merge(df1, df2, on='Testfile', how='outer')
    
    print("Merge completed!")
    print(f"Total unique files: {len(merged_df)}")
    merged_df.to_csv('upstream.csv', index=False)

    import pdb
    pdb.set_trace()
    df_local = pd.concat([df3, df4], axis=0, ignore_index=True)
    df_local = df_local.rename(columns={df_local.columns[0]: 'Testfile', df_local.columns[1]: 'xpu_ops-total', df_local.columns[2]: 'xpu_ops-PASSED', df_local.columns[3]: 'xpu_ops-Skipped', df_local.columns[4]: 'xpu_ops-Failed'})

    df_local["Testfile"] = df_local["Testfile"].astype(str)
    merged_df["Testfile"] = merged_df["Testfile"].astype(str)
    import pdb
    pdb.set_trace()

    
    merged_df = pd.merge(merged_df, df_local, on='Testfile', how='outer')

    # Fill NaN values with 0 and convert to integers
    for col in merged_df.columns:
        if col != "Testfile":
            merged_df[col] = merged_df[col].fillna(0).astype(int)
 
    merged_df.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge two .csv files of different workflow')
    parser.add_argument('-o', '--output', default='merged.csv', help='The output csv file (default: merged.csv)')

    args = parser.parse_args()

    merge(args.output)
 
