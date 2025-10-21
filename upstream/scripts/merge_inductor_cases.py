import os
import argparse
import pandas as pd

def merge(output: str):
    # Read all 4 CSV files
    #df1 = pd.read_csv('cases/details_rocm.csv', delimiter='|', encoding='utf-8')
    df1 = pd.read_csv('cases/details_rocm.csv', delimiter='|')
    #df2 = pd.read_csv('cases/details_xpu.csv', delimiter='|', encoding='utf-8') 
    df2 = pd.read_csv('cases/details_xpu.csv', delimiter='|') 
 
    df1 = df1.rename(columns={df1.columns[0]: 'Testfile', df1.columns[1]: 'Class_unified', df1.columns[2]: 'Testcase_unified', df1.columns[3]: 'Class', df1.columns[4]: 'Testcase', df1.columns[5]: 'Result', df1.columns[6]: 'SkipReason'})
    df2 = df2.rename(columns={df2.columns[0]: 'Testfile', df2.columns[1]: 'Class_unified', df2.columns[2]: 'Testcase_unified', df2.columns[3]: 'Class-XPU', df2.columns[4]: 'Testcase-XPU', df2.columns[5]: 'Result-XPU', df2.columns[6]: 'SkipReason-XPU'})

    # Merge sequentially
    merged_df = pd.merge(df1, df2, on=['Testfile', 'Class_unified', 'Testcase_unified'], how='outer')

    substring = 'inductor'
    merged_df = merged_df[merged_df.iloc[:, 0].str.contains(substring, na=False)]
    
    print("Merge completed!")
    print(f"Total unique files: {len(merged_df)}")
    merged_df.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge two .csv files of different workflow')
    parser.add_argument('-o', '--output', default='merged_details.csv', help='The output csv file (default: merged.csv)')

    args = parser.parse_args()

    merge(args.output)
 
