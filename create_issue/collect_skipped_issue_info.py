import os
from attr import dataclass
import pandas as pd
import argparse
from get_duplicated_issues.get_duplicated_issues import download_all_open_issues_and_get_issue_with_label

def main():
    artifacts = args.artifacts
    xpu_issues_folder = os.path.curdir + "/xpu_issue"
    if os.path.exists(xpu_issues_folder) is False:
        os.mkdir(xpu_issues_folder)
    # If check known issues download xpu-ops issues first
    skip_issues = download_all_open_issues_and_get_issue_with_label("intel/torch-xpu-ops", os.getenv("GITHUB_TOKEN"), xpu_issues_folder)

    ut_failure_list = {}
    with open('ut_failure_list.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("| op_ut | third_party.torch-xpu-ops.test.xpu."):
                items = line.split("|")
                try:
                    test_file = "/".join(items[2].strip().split(".")[4:-1]) + ".py"
                    test_class = items[2].strip().split(".")[-1]
                    test_case = items[3].strip()
                    error_message = items[5].strip()
                    ut_failure_list[f"{test_file}|{test_class}|{test_case}"] = error_message
                except:
                    error_message = ""
    passed_ut = []
    with open("./passed_op_ut.log", "r") as f:
        lines = f.readlines()
        passed_ut = [line.strip() for line in lines]
          
    dynamic_skiped1 = []

    def match_test_case(test_file: str, test_class: str, test_case:str, passed_ut: list):
        for line in passed_ut:
            if test_file in line and test_class in line and test_case in line:
                return True
        return False

    def get_trace(xml_file: str, test_case: str, pytest_command: str):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()
            trace = ""
            for testcase in root.iter('testcase'):
                if testcase.get('name') == test_case:
                    error = testcase.find('error')
                    failure = testcase.find('failure')
                    if error is not None:
                        trace = f"\n```\nCommand: {pytest_command}\n{error.text}\n```"
                    elif failure is not None:
                        trace = f"\n```\nCommand: {pytest_command}\n{failure.text}\n```"
                    else:
                        trace = ''
                    break
            return trace
        except Exception:
            print(f"Failed to parse XML file: {xml_file} for test case: {test_case}")
            return ''

    def get_result(case):
        from junitparser import JUnitXml, Error, Failure, Skipped
        if isinstance(case, dict):
            return case.get('status', 'failed')

        result = "passed"
        if case.result:
            if isinstance(case.result[0], Error):
                result = "error"
            elif isinstance(case.result[0], Skipped):
                result = "skipped"
            elif isinstance(case.result[0], Failure):
                result = "failed"
        return result

    def get_case_result(test_xml, test_case: str):
        from junitparser import JUnitXml

        try:
            xml = JUnitXml.fromfile(test_xml)
            for suite in xml:
                for case in suite:                    
                    if case.name == test_case:
                        return get_result(case)
            return "not found the case"
        except Exception as e:
            print(f"Error parsing XML file {test_xml}: {e}")
            return "xml parse error" 
            
    if not args.merge_only:
        for skip_issue, owner, labels in skip_issues:   
            
            print(f"Skipped issue file: {skip_issue}")
            
            with open(f"{xpu_issues_folder}/{skip_issue}", "r", encoding="utf-8") as f:
                content = f.read()
                error_type = ""
                if "NotImplementedError" in content:
                    error_type = "Feature gap"
                elif "AssertionError: Torch not compiled with CUDA enabled" in content:
                    error_type = "Test case to be enabled"
                elif "AssertionError" in content or "RuntimeError" in content or "TypeError" in content or "ValueError" in content:
                    error_type = "Failure(xpu broken)"
                else:
                    error_type = ""

                error_type = "dynamic_skip"
                start = False
                for line in content.splitlines():
                    if line.endswith("Cases:"):
                        start = True
                    elif start == True and line.startswith("~~op_ut,third_party.torch-xpu-ops.test.xpu."):
                        continue
                    elif start == True and line.startswith("op_ut,third_party.torch-xpu-ops.test.xpu.") or line.startswith("op_ut,test."):
                        items = line.split(",")                
                        
                        stock = True if line.startswith("op_ut,test.") else False

                        try:
                            test_file = "/".join(items[1].split(".")[4:-1]) + ".py" if not stock else "/".join(items[1].split(".")[1:-1]) + ".py"
                            test_case = items[2].strip()
                            test_class = items[1].split(".")[-1]
                            
                            test_xml = f"{artifacts}/op_ut_with_skip." + test_file.replace(".py", ".xml").replace("/", ".")
                            if test_xml is not None and not os.path.exists(test_xml):
                                test_xml = f"{artifacts}/op_ut_with_all." + test_file.replace(".py", ".xml").replace("/", ".")                            
                                if test_xml is not None and not os.path.exists(test_xml):
                                    test_xml = None
                            if test_xml is not None:
                                trace = get_trace(test_xml, test_case, f"pytest -m xpu {test_file} -k {test_case}")

                            error_message = ut_failure_list.get(f"{test_file}|{test_class}|{test_case}", "")
                            if len(error_message) == 0:
                                error_message = ut_failure_list.get(f"{test_file.replace('_xpu', '')}|{test_class}|{test_case}", "")
                                if len(error_message) == 0:
                                        
                                    # if match_test_case(test_file.replace('.py','').replace('/', '.'), test_class, test_case, passed_ut):
                                    #     print(f"{test_file.replace('.py','')},{test_class},{test_case} is in passed_ut")
                                    #     error_message = "Passed"
                                    # else:
                                    if test_xml is not None:
                                        case_result = get_case_result(test_xml, test_case)
                                        #print(f"Cannot find error message for op_ut,third_party.torch-xpu-ops.test.xpu.{test_file.replace('.py','')},{test_class},{test_case}, marking as not in pass list and failure list")
                                        error_message = f"{case_result}"

                                        if error_message in ["not found the case", "xml parse error"]:
                                            print(f"grep {test_case} {test_xml}")
                                    else:
                                        print(f"xml file is None for {test_file}, {test_case}")
                                        error_message = f"no xml file"
                                        
                            
                            # Escape quotes and newlines for CSV compatibility
                            escaped_error_message = error_message.replace('"', '""').replace('\n', '\\n')
                            escaped_trace = trace.replace('"', '""').replace('\n', '\\n')
                            
                            _test_file = f"test/xpu/{test_file}" if not stock else f"test/{test_file}"
                            updated = False
                            for i, line in enumerate(dynamic_skiped1):
                                if f"{_test_file}|{items[1].split('.')[-1]}|{test_case}" in line:
                                    issue_ids = f"{line.split('|')[4]},issues/{skip_issue.replace('.txt', '')}<{owner}>"
                                    labels = f"{line.split('|')[6]},{labels}"
                                    labels = ",".join(sorted(set(labels.split(','))))                                
                                    dynamic_skiped1[i] = f"{_test_file}|{items[1].split('.')[-1]}|{test_case}|{error_type}|{issue_ids}|{stock}|\"{escaped_error_message}\"|{labels}|\"{escaped_trace}\""
                                    
                                    updated = True
                                    break
                            
                            if not updated:
                                dynamic_skiped1.append(f"{_test_file}|{items[1].split('.')[-1]}|{test_case}|{error_type}|issues/{skip_issue.replace('.txt', '')}<{owner}>|{stock}|\"{escaped_error_message}\"|{labels}|\"{escaped_trace}\"")
                                
                        except Exception as e:
                            print(f"Error processing line: {line} in issue {skip_issue}, error: {e}")
                    else:
                        start = False
    
        with open("dynamic_skipped_list_xpuindex.csv", "w") as f:
            for item in dynamic_skiped1:
                f.write(item + "\n")
    
    def merge_skip_lists(xlsx_file1: str, csv_file2: str, output_xlsx: str):
        df1 = pd.read_excel(xlsx_file1, sheet_name='dynamic_skipped_list_xpuindex')
        df2 = pd.read_csv(csv_file2, sep='|', header=None, names=["File", "Class", "Case", "Reason", "Issue Link", "Stock Issue", "Error type", "Labels", "Trace"])
        
        # Perform outer merge based on test_file, test_class, and test_case
        merged_df = pd.merge(
            df1, 
            df2, 
            on=["File", "Class", "Case"], 
            how="outer",
            suffixes=('_old', '_new')
        )
        
        # Optionally, combine columns from both sources (keeping new values when available)
        for col in df2.columns:
            if col not in ["test_file", "test_class", "test_case"]:
                old_col = f"{col}_old"
                new_col = f"{col}_new"
                if old_col in merged_df.columns and new_col in merged_df.columns:
                    merged_df[col] = merged_df[new_col].fillna(merged_df[old_col])
                    merged_df.drop([old_col, new_col], axis=1, inplace=True)

        merged_df.to_excel(output_xlsx, index=False)

    if os.path.exists("dynamic_skipped_list_xpuindex.xlsx") and os.path.exists("dynamic_skipped_list_xpuindex.csv"):
        merge_skip_lists("dynamic_skipped_list_xpuindex.xlsx", "dynamic_skipped_list_xpuindex.csv", "dynamic_skipped_list_xpuindex_merged.xlsx")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GitHub issues from UT failure list")
    parser.add_argument('--artifacts', type=str, default='./nightly/artifacts', help="nightly artifacts")
    parser.add_argument('--merge_only', action='store_true', default=False, help="only mernge the csv to xlsx")
    args = parser.parse_args()
    main()
