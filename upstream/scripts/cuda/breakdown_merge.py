import pandas as pd
import numpy as np

def merge_breakdown(skipped_csv: str, xlsx_path: str, output: str):
    # Load the CSV files into pandas DataFrames
    df_a = pd.read_csv(skipped_csv, delimiter=',', engine='python')
    df_b = pd.read_excel(xlsx_path, sheet_name='Cuda pass xpu skip')
    df_c = pd.read_excel(xlsx_path, sheet_name='to_be_enabled')

    df_a = df_a[["Testfile", "Class_unified", "Testcase_unified"]]
    df_b = df_b[["Testfile", "Class_unified", "Testcase_unified", "Reason", "detail reason", "status"]]
    df_c = df_c[["Testfile", "Class_unified", "Testcase_unified", "Reason", "detail reason", "status"]]

    combine_df = pd.concat([df_b, df_c], ignore_index=True)

    df_a['Class_unified'] = df_a['Class_unified'].str.replace('cuda', 'gpu')
    df_a['Testcase_unified'] = df_a['Testcase_unified'].str.replace('cuda', 'gpu')
    combine_df['Class_unified'] = combine_df['Class_unified'].str.replace('cuda', 'gpu')
    combine_df['Testcase_unified'] = combine_df['Testcase_unified'].str.replace('cuda', 'gpu')

    merged_df = pd.merge(df_a, combine_df, on=["Testfile", "Class_unified", "Testcase_unified"], how='left')

    merged_df.to_csv(output, index=False)


merge_breakdown("./xpu_skipped_inductor.csv", "./Inductor_ut_status_ww46.xlsx", "./inductor_merged.csv")

def merge_breakdown_noninductor(skipped_csv: str, xlsx_path: str, output: str):
    # Load the CSV files into pandas DataFrames
    df_a = pd.read_csv(skipped_csv, delimiter=',', engine='python')
    df_b = pd.read_excel(xlsx_path, sheet_name='Non-Inductor XPU Skip')

    df_a = df_a[["Testfile", "Class_unified", "Testcase_unified"]]
    df_b = df_b[["Testfile", "Class_unified", "Testcase_unified", "Reason", "DetailReason"]]

    df_a['Class_unified'] = df_a['Class_unified'].str.replace('cuda', 'gpu')
    df_a['Testcase_unified'] = df_a['Testcase_unified'].str.replace('cuda', 'gpu')
    df_b['Class_unified'] = df_b['Class_unified'].str.replace('cuda', 'gpu')
    df_b['Testcase_unified'] = df_b['Testcase_unified'].str.replace('cuda', 'gpu')

    merged_df = pd.merge(df_a, df_b, on=["Testfile", "Class_unified", "Testcase_unified"], how='left')

    merged_df.to_csv(output, index=False)


merge_breakdown_noninductor("./xpu_skipped_all.csv", "./Non_inductor_ut_status_ww45.xlsx", "./noninductor_merged.csv")
