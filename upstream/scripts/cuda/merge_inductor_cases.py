import os
import argparse
import pandas as pd

def merge(f1: str, f2: str, output: str):
    # Read all 4 CSV files
    #df1 = pd.read_csv('cases/details_rocm.csv', delimiter='|', encoding='utf-8')
    df1 = pd.read_csv(f1, delimiter='|')
    #df2 = pd.read_csv('cases/details_xpu.csv', delimiter='|', encoding='utf-8') 
    df2 = pd.read_csv(f2, delimiter='|') 

    if len(df1) > 0 and df1.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df1 = df1.iloc[:-1]
    if len(df2) > 0 and df2.iloc[-1, 1] == "**Total**" :  # iloc[-1, 1] gets last row, 2nd column
        df2 = df2.iloc[:-1]

    df1 = df1.rename(columns={df1.columns[0]: 'Testfile', df1.columns[1]: 'Class_unified', df1.columns[2]: 'Testcase_unified', df1.columns[3]: 'Class', df1.columns[4]: 'Testcase', df1.columns[5]: 'Result', df1.columns[6]: 'SkipReason'})
    df2 = df2.rename(columns={df2.columns[0]: 'Testfile', df2.columns[1]: 'Class_unified', df2.columns[2]: 'Testcase_unified', df2.columns[3]: 'Class-XPU', df2.columns[4]: 'Testcase-XPU', df2.columns[5]: 'Result-XPU', df2.columns[6]: 'SkipReason-XPU'})

    # Merge sequentially
    merged_df = pd.merge(df1, df2, on=['Testfile', 'Class_unified', 'Testcase_unified'], how='outer')

    substring = '/inductor/'
    merged_df = merged_df[merged_df.iloc[:, 0].str.contains(substring, na=False)]
    
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
    split_with_progress(merged_df, 4, output)
    
    mask = (merged_df['Result'] == 'passed') & (merged_df['Result-XPU'].isin(['skipped', '']) | merged_df['Result-XPU'].isna())
    filtered_df = merged_df[mask]
    filtered_df.to_csv("xpu_skipped_inductor.csv", index=False)

    for index, row in filtered_df.iterrows():
        testfile = row['Testfile']
        class_xpu = row['Class-XPU']
        testcase_xpu = row['Testcase-XPU'] 
        testcase = row['Testcase'].replace("cuda", "xpu") 

        with open('xpu_skipped_inductor.txt', 'a') as file: 
            file.write(f"pytest -v {testfile} -k {testcase} \n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge two .csv files of different workflow')
    parser.add_argument('--f1', default='data/cases/details_cuda.csv', help='The reference input csv file (default: data/cases/details_cuda.csv)')
    parser.add_argument('--f2', default='data/cases/details_stock_xpu.csv', help='The xpu input csv file (default: data/cases/details_stock_xpu.csv)')
    parser.add_argument('-o', '--output', default='merged_details.csv', help='The output csv file (default: merged.csv)')

    args = parser.parse_args()

    merge(args.f1, args.f2, args.output)
 
