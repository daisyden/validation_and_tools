import os
from attr import dataclass
import pandas as pd
import argparse
from get_duplicated_issues.get_duplicated_issues import download_all_open_issues_and_get_skiplist

def main():

    xpu_issues_folder = os.path.curdir + "/xpu_issue"
    if os.path.exists(xpu_issues_folder) is False:
        os.mkdir(xpu_issues_folder)
    # If check known issues download xpu-ops issues first
    skip_issues = download_all_open_issues_and_get_skiplist("intel/torch-xpu-ops", os.getenv("GITHUB_TOKEN"), xpu_issues_folder)

    dynamic_skiped = []
    dynamic_skiped2 = []
    dynamic_skiped3 = []
    for skip_issue in skip_issues:
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
            for line in content.splitlines():
                if line.startswith("op_ut,third_party.torch-xpu-ops.test.xpu."):
                    items = line.split(",")
                    
                    try:
                        test_file = "/".join(items[1].split(".")[4:-1]) + ".py"
                        test_case = items[2].strip()
                        if len((f"{test_file}|{test_case}").split("|")) != 2:
                            print(f"Invalid skip item: {f'{test_file}|{test_case}'}, skipping...")
                            import pdb
                            pdb.set_trace()
                        dynamic_skiped.append(f"{test_file}|{test_case}")
                
                        dynamic_skiped2.append(f"test/xpu/{test_file},{items[1].split('.')[-1]},{test_case},{error_type},https://github.com/intel/torch-xpu-ops/issues/{skip_issue.replace('.txt', '')}")
                        dynamic_skiped3.append(f"test/{test_file.replace('_xpu', '')},{items[1].split('.')[-1].replace('XPU', 'CUDA')},{test_case.replace('xpu', 'cuda')},{error_type},https://github.com/intel/torch-xpu-ops/issues/{skip_issue.replace('.txt', '')}")
                    except Exception as e:
                        import pdb
                        pdb.set_trace()
                        print(f"Error processing line: {line} in issue {skip_issue}, error: {e}")
    with open("dynamic_skipped_list_xpuindex.csv", "w") as f:
        for item in dynamic_skiped2:
            f.write(item + "\n")

    with open("dynamic_skipped_list_cudaindex.csv", "w") as f:
        for item in dynamic_skiped2:
            f.write(item + "\n")

    passed = []
    with open("skipped/Inductor-XPU-UT-Data-78622166d521bbb8c070dd0e3735d90529663cda-skipped_ut-20100148130-1/skipped_ut/passed_op_ut.log", "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("op_ut,third_party.torch-xpu-ops.test.xpu."):
                items = line.split(",")
                try:
                    test_file = "/".join(items[1].split(".")[4:-1]) + ".py"
                    test_case = items[2].strip()
                    if len((f"{test_file}|{test_case}").split("|")) != 2:
                        print(f"Invalid skip item: {f'{test_file}|{test_case}'}, skipping...")
                        import pdb
                        pdb.set_trace()
                    passed.append(f"{test_file}|{test_case}")
                except Exception as e:
                    import pdb
                    pdb.set_trace()
                    print(f"Error processing line: {line} in passed_ut.log, error: {e}")

    failed = []
    with open("ut_failure_list.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("| op_ut | third_party.torch-xpu-ops.test.xpu."):
                items = line.split("|")
                try:
    
                    test_file = "/".join(items[2].strip().split(".")[4:-1]) + ".py"
                    test_case = items[3].strip()
                    if len((f"{test_file}|{test_case}").split("|")) != 2:
                        print(f"Invalid skip item: {f'{test_file}|{test_case}'}, skipping...")
                        import pdb
                        pdb.set_trace()
                    if f"{test_file}|{test_case}" not in failed:
                        failed.append(f"{test_file}|{test_case}")
                except Exception as e:
                    import pdb
                    pdb.set_trace()
                    print(f"Error processing line: {line} in ut_failure_list.csv, error: {e}")

    
    from skip_list_common import skip_dict

    total = 0
    for key in skip_dict.keys():
        if skip_dict[key] is not None:
            total += len(skip_dict[key])
    print(f"Original skip_dict has {len(skip_dict)} test files with total {total} test cases.")

    for key in skip_dict.keys():
        if skip_dict[key] is not None and len(skip_dict[key]) != 0:
            cases = skip_dict[key]
            for case in cases:
                if (case.startswith('test')) and (f"{key}|{case}" not in failed):
                    if f"{key}|{case}" not in passed:
                        print(f"Adding {key}|{case} to dynamic skipped list")
                        passed.append(f"{key}|{case}")       
   
    for item in dynamic_skiped + passed:
        if len(item.split("|")) != 2:
            print(f"Invalid skip item: {item}, skipping...")
            import pdb
            pdb.set_trace()
        test_file, test_case = item.split("|")
        if "test_convolution_xpu.py" in test_file:
            import pdb
            pdb.set_trace
        if test_file in skip_dict.keys():
            if skip_dict[test_file] is not None and test_case in skip_dict[test_file]:
                #skip_dict[test_file] = tuple(list(skip_dict[test_file]).remove(test_case))
                new_tuple = tuple(x for x in skip_dict[test_file] if x != test_case)
                print(f"Removing {test_case} from skip_dict[{test_file}], {len(skip_dict[test_file])} -> {len(new_tuple)}")
                if len(new_tuple) == 0:
                    skip_dict[test_file] = None
                else:
                    skip_dict[test_file] = new_tuple

    
    import json
    print(json.dumps(skip_dict, indent=4, sort_keys=True)) 

    with open('skip_list_common2.json', 'w') as f:
        json.dump(skip_dict, f, indent=4, sort_keys=True)   

    new_total = 0
    for key in skip_dict.keys():
        if skip_dict[key] is not None:
            new_total += len(skip_dict[key])

    print(f"New skip_dict has {len(skip_dict)} test files with total {new_total} test cases. original is {total}. passed count is {len(passed)}. dynamically skipped count is {len(dynamic_skiped)}")

    udpated_list = []
    with open('ut_failure_list.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("| op_ut | third_party.torch-xpu-ops.test.xpu."):
                items = line.split("|")
                try:
                    #import pdb; pdb.set_trace()
                    test_file = "/".join(items[2].strip().split(".")[4:-1]) + ".py"
                    test_case = items[3].strip()
                    if test_file in skip_dict.keys():
                        if skip_dict[test_file] != None and test_case in skip_dict[test_file]:
                            udpated_list.append(line)                              
                except Exception as e:
                    print(f"Error processing line: {line} in ut_failure_list.csv, error: {e}")
                    import pdb
                    pdb.set_trace()

    with open('ut_failure_list_updated.csv', 'w') as f:
        for line in udpated_list:
            f.write(line) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GitHub issues from UT failure list")
    parser.add_argument('--skiplist', type=str, default=False, help="skip list")
    args = parser.parse_args()
    main()
