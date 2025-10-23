import os
import argparse
import pandas as pd

def merge(output: str):
    # Read all 4 CSV files
    df1 = pd.read_csv('slow.csv')
    df2 = pd.read_csv('periodic.csv')
    df3 = pd.read_csv('b200.csv')
    df4 = pd.read_csv('inductor-unittest.csv')
    
    df_xpu = pd.read_csv('stock_pytorch_xpu.csv')
    
    # Rename columns to avoid conflicts
    df1 = df1.rename(columns={df1.columns[0]: 'file', df1.columns[1]: 'slow'})
    df2 = df2.rename(columns={df2.columns[0]: 'file', df2.columns[1]: 'periodic'})
    df3 = df3.rename(columns={df3.columns[0]: 'file', df3.columns[1]: 'b200'})
    df4 = df4.rename(columns={df4.columns[0]: 'file', df4.columns[1]: 'inductor-unittest'})
    
    # Merge sequentially
    merged_df = pd.merge(df1, df2, on='file', how='outer')
    merged_df = pd.merge(merged_df, df3, on='file', how='outer')
    merged_df = pd.merge(merged_df, df4, on='file', how='outer')
    
    # Fill NaN values with 0 and convert to integers
    for col in ['slow', 'periodic', 'b200', 'inductor-unittest']:
        merged_df[col] = merged_df[col].fillna(0).astype(int)
    
    print("Merge completed!")
    print(f"Total unique files: {len(merged_df)}")
    merged_df.to_csv('cuda_merged.csv', index=False)

    df_xpu = pd.read_csv('stock_pytorch_xpu.csv')
    df_xpu = df_xpu.rename(columns={df_xpu.columns[0]: 'file', df_xpu.columns[1]: 'xpu'})
    merged_df = pd.merge(merged_df, df_xpu, on='file', how='outer')

    df5 = pd.read_csv("torch-xpu-ops/summary_parsed.csv")
    df6 = pd.read_csv("distributed_weekly/dist_summary_parsed.csv")

    df5 = df5.drop(['UT', 'Errors', 'Source'], axis=1)
    df6 = df6.drop([' Category ', ' UT '], axis=1)

    df5 = df5.rename(columns={df5.columns[0]: 'file', df5.columns[1]: 'total', df5.columns[2]: 'passed', df5.columns[3]: 'skipped', df5.columns[4]: 'failed'})
    df6 = df6.rename(columns={df6.columns[0]: 'file', df6.columns[1]: 'total', df6.columns[2]: 'passed', df6.columns[3]: 'skipped', df6.columns[4]: 'failed'})

    ops_df = pd.concat([df5, df6], ignore_index=True)
    merged_df = pd.merge(merged_df, ops_df, on='file', how='outer')

    merged_df.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge two .csv files of different workflow')
    parser.add_argument('-o', '--output', default='merged.csv', help='The output csv file (default: merged.csv)')

    args = parser.parse_args()

    merge(args.output)
 
