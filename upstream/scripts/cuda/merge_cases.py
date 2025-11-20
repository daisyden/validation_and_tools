import os
import argparse
import pandas as pd

def merge(f1: str, f2: str, f3: str, f4: str, output: str, matching: str, notmatch: str):
    # Read all 4 CSV files
    #df1 = pd.read_csv('cases/details_rocm.csv', delimiter='|', encoding='utf-8')
    df1 = pd.read_csv(f1, delimiter='|')
    #df2 = pd.read_csv('cases/details_xpu.csv', delimiter='|', encoding='utf-8') 
    df2 = pd.read_csv(f2, delimiter='|') 
    if f3 is not None:
        df3 = pd.read_csv(f3, delimiter='|') 
    if f4 is not None:
        df4 = pd.read_csv(f4, delimiter='|') 


    if len(df1) > 0 and df1.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df1 = df1.iloc[:-1]
    if len(df2) > 0 and df2.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df2 = df2.iloc[:-1]
    if df3 is not None and len(df3) > 0 and df3.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df3 = df3.iloc[:-1]
    if df4 is not None and len(df4) > 0 and df4.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df4 = df4.iloc[:-1]


    df1 = df1.rename(columns={df1.columns[0]: 'Testfile', df1.columns[1]: 'Class_unified', df1.columns[2]: 'Testcase_unified', df1.columns[3]: 'Class', df1.columns[4]: 'Testcase', df1.columns[5]: 'Result', df1.columns[6]: 'SkipReason'})
    df2 = df2.rename(columns={df2.columns[0]: 'Testfile', df2.columns[1]: 'Class_unified', df2.columns[2]: 'Testcase_unified', df2.columns[3]: 'Class-stock-xpu', df2.columns[4]: 'Testcase-stock-xpu', df2.columns[5]: 'Result-stock-xpu', df2.columns[6]: 'SkipReason-stock-xpu'})
    # import pdb
    # pdb.set_trace()
    df3 = df3.rename(columns={df3.columns[0]: 'Testfile', df3.columns[1]: 'Class_unified', df3.columns[2]: 'Testcase_unified', df3.columns[3]: 'Class-xpu-ops', df3.columns[4]: 'Testcase-xpu-ops', df3.columns[5]: 'Result-xpu-ops', df3.columns[6]: 'SkipReason-xpu-ops'})
    df4 = df4.rename(columns={df4.columns[0]: 'Testfile', df4.columns[1]: 'Class_unified', df4.columns[2]: 'Testcase_unified', df4.columns[3]: 'Class-xpu-ops', df4.columns[4]: 'Testcase-xpu-ops', df4.columns[5]: 'Result-xpu-ops', df4.columns[6]: 'SkipReason-xpu-ops'})
    #df3['Class_unified'] = df3['Class_unified'].str.split('.').str[-1]  

    df3 = pd.concat([df3, df4], axis=0, ignore_index=True)

    # Merge sequentially
    merged_df = pd.merge(df1, df2, on=['Testfile', 'Class_unified', 'Testcase_unified'], how='outer')
    merged_df = pd.merge(merged_df, df3, on=['Testfile', 'Class_unified', 'Testcase_unified'], how='outer')

    # merged_df['xpu-ops tested'] = merged_df['Testfile'].isin(df3['Testfile']).map({True: 'yes', False: 'no'})
    #merged_df['owern'] = ('test/distributed' in merged_df['Testfile']).map({True: 'Cherry', False: ''})


    substring = matching 
    if substring is not None and len(substring) > 0:
        merged_df = merged_df[merged_df.iloc[:, 0].str.contains(substring, na=False)]
    
    substring = notmatch
    if substring is not None and len(substring) > 0:
        merged_df = merged_df[~merged_df.iloc[:, 0].str.contains(substring, na=False)]

    print("Merge completed!")
    print(f"Total unique files: {len(merged_df)}")
    #merged_df.to_csv(output, index=False)


    from tqdm import tqdm
    import numpy as np

    def split_with_progress(df, num_parts, base_filename):
        chunks = np.array_split(df, num_parts)
        
        for i, chunk in tqdm(enumerate(chunks), total=num_parts, desc="Writing files"):
            filename = f"{output}_part_{i+1:03d}.csv"
            chunk.to_csv(filename, index=False)
        
        print(f"Successfully created {num_parts} files")

    # Usage
    # split_with_progress(merged_df, 4, output)
    merged_df.to_csv(output, index=False)

    #mask = merged_df['Result-XPU'].isin(['skipped', '']) | merged_df['Result-XPU'].isna()
    condition1 = (merged_df['Result-stock-xpu'] != 'passed') | (merged_df['Result-stock-xpu'].isna())
    condition2 = (merged_df['Result-xpu-ops'] != 'passed') | (merged_df['Result-xpu-ops'].isna())
    mask = condition1 & condition2
    filtered_df = merged_df[mask]
    filtered_df.to_csv(f"xpu_all_skipped.csv", index=False)

    condition1 = merged_df['Result'] == 'passed'
    condition2 = (merged_df['Result-stock-xpu'] != 'passed') | merged_df['Result-stock-xpu'].isna()
    condition3 = (merged_df['Result-xpu-ops'] != 'passed') | merged_df['Result-xpu-ops'].isna()
    mask = condition1 & condition2 & condition3
    filtered_df = merged_df[mask]
    filtered_df.to_csv("xpu_only_skipped.csv", index=False)

    #for index, row in filtered_df.iterrows():
    #    testfile = row['Testfile']
    #    class_xpu = row['Class-XPU']
    #    testcase_xpu = row['Testcase-XPU'] 
    #    testcase = row['Testcase'].replace("cuda", "xpu") 

    #    with open('xpu_skipped.txt', 'a') as file: 
    #        file.write(f"pytest -v {testfile} -k {testcase} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge two .csv files of different workflow')
    parser.add_argument('--f1', default='data/cases/details_cuda.csv', help='The reference input csv file (default: data/cases/details_cuda.csv)')
    parser.add_argument('--f2', default='data/cases/details_stock_xpu.csv', help='The xpu input csv file (default: data/cases/details_stock_xpu.csv)')
    parser.add_argument('--f3', default='data/cases/details_xpu-ops.csv', help='The xpu input csv file (default: data/cases/details_xpu-ops.csv)')
    parser.add_argument('--f4', default='data/cases/details_distributed.csv', help='The xpu distributed input csv file (default: data/cases/details_distributed.csv)')
    parser.add_argument('--match', help='The path to filter')
    parser.add_argument('--notmatch', help='The path to skip')
    parser.add_argument('-o', '--output', default='merged_details.csv', help='The output csv file (default: merged.csv)')

    args = parser.parse_args()

    merge(args.f1, args.f2, args.f3, args.f4, args.output, args.match, args.notmatch)
 
