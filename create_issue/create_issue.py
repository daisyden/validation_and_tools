import os
from attr import dataclass
import pandas as pd
from get_duplicated_issues.get_duplicated_issues import get_duplicated_issues, download_all_open_issues_and_get_skiplist
import argparse
from get_duplicated_issues.github_issue import Github_Issue

@dataclass
class FailureGroup:
    id: int
    error_msg: str
    skipped: list[str]
    commands: list[str]
    tracees: list[str]
    duplicated: list[str]
    new_duplicated: list[str]

def write_issue_file(failure_group: FailureGroup, error_message: str, issue_folder: str, submit=False, skiplist=None):
    filename = f'{issue_folder}/issue_group{failure_group.id}.txt'
    with open(filename, 'w') as f:
        if "test/xpu" in failure_group.skipped[0] or "xpu.py" in failure_group.skipped[0]:
            title = f"[ut] {error_message[:min(100, len(error_message))]}\n"
        else:
            title = f"[upstream_ut] {error_message[:min(100, len(error_message))]}\n"
    
        f.write("Title: " + title)

        content = ""
        cases = '\n'.join(failure_group.skipped)
        content += f"\nCases:\n{cases}\n"
        f.write(f"Cases:\n{cases}\n")
        
        commands_str = '\n'.join(failure_group.commands)
        content += f"\npytest_command:\n{commands_str}\n"
        f.write(f"\npytest_command:\n{commands_str}\n")

        content += f"\nError Message:\n{error_message}\n"
        f.write(f"\nError Message:\n{error_message}\n")
        
        f.write("\nTrace Example:\n")
        if len(failure_group.tracees) > 0:
            content += "\nTrace Example:\n" +  failure_group.tracees[-1]
            f.write(failure_group.tracees[-1])

        if submit:
            gh = Github_Issue("intel/torch-xpu-ops", os.getenv("GITHUB_TOKEN"))
            print(f"Wrote merged issue file: {filename}")
            issue_id = gh.create_issue_with_label(title, body=content, labels=["skipped"])

        if skiplist is not None:
            f.write("[xpu_triage_bot]: Skiplist comments:\n")
            comments = []
            for case in cases.split('\n'):
                print(case)
                test_class = case.split(',')[1]
                test_file = test_class.split('.')[-2] + '.py'
                test_case = case.split(',')[2]
                skip_reason_row = skiplist[(skiplist['File'] == test_file) & (skiplist['Testcase'] == test_case)]
                if skip_reason_row is not None and len(skip_reason_row['Skipreason'].values) > 0:
                    comments.append(f"{test_file}|{test_case}|{skip_reason_row['Skipreason'].values[0]}")                
            if submit:
                gh.add_comment(f"\n\n[xpu_triage_bot]:\n Skiplist comments:\n" + '\n'.join(comments), issue_id)
            else:
                f.write('\n'.join(comments))
    print(f"Wrote issue file: {filename}")
    return issue_id if submit else -1


def get_duplicated_known_issues(failure_group: FailureGroup, error_message: str, issue_folder: str, ratio: float = 0.7):
    # Pattern match for existing GitHub issues
    print(f"\n# Checking known duplicated issues for group {failure_group.id}...")
    duplicated_issue = get_duplicated_issues(failure_group.id, failure_group.skipped, error_message, failure_group.tracees[-1], issue_folder, ratio)
    failure_group.duplicated = list(set([ _issue for _issue in duplicated_issue])) if duplicated_issue is not None else []
    print(f"(Patern match) Known duplicated issues for group {failure_group.id}: {failure_group.duplicated}")

    if len(failure_group.duplicated) == 0 and args.ai:
        from get_duplicated_issues.get_duplicated_issues_ai import get_duplicated_issues_with_rag
        duplicated_issue = get_duplicated_issues_with_rag(failure_group.skipped, error_message, failure_group.tracees[-1], issue_folder)
        failure_group.duplicated = list(set([ _issue["url"] for _issue in duplicated_issue]))
        print(f"(AI) Known duplicated issues for group {failure_group.id}: {failure_group.duplicated}")
    return failure_group.duplicated


def get_duplicated_new_issues(failure_group: FailureGroup, error_message: str, issue_folder: str, ratio: float = 0.7):
    # Pattern match for issue detected in this script
    print(f"\n# Checking new duplicated issues for group {failure_group.id}...")
    new_duplicated_issue = get_duplicated_issues(failure_group.id, failure_group.skipped, error_message, failure_group.tracees[-1], issue_folder, ratio)
    failure_group.new_duplicated = list(set([ _issue for _issue in new_duplicated_issue])) if new_duplicated_issue is not None else []
    print(f"(Patern match) New duplicated issues for group {failure_group.id}: {failure_group.new_duplicated}")

    if len(failure_group.new_duplicated) == 0 and args.ai:
        from get_duplicated_issues.get_duplicated_issues_ai import get_duplicated_issues_with_rag
        __new_duplicated_issue = get_duplicated_issues_with_rag(failure_group.skipped, error_message, failure_group.tracees[-1], issue_folder)
        failure_group.new_duplicated = list(set([ os.patch.basename(_issue["url"]) for _issue in 
                                       ( __new_duplicated_issue["high_confidence"] + 
                                        __new_duplicated_issue["medium_confidence"])]))
        print(f"(AI) New duplicated issues for group {failure_group.id}: {failure_group.new_duplicated}")
    return failure_group.new_duplicated


def merge_failure_groups(failure_group_list: list[FailureGroup]) -> list[FailureGroup]:
    import numpy as np
    n_groups = len(failure_group_list)
    relationship_matrix = np.zeros((n_groups, n_groups), dtype=int)

    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                relationship_matrix[i][j] = 1
            else:
                # Mark 1 if i and j are duplicated issues with each other
                if f"issue_group{i}" in failure_group_list[j].new_duplicated or f"issue_group{j}" in failure_group_list[i].new_duplicated:
                    relationship_matrix[i][j] = 1
                else:
                    relationship_matrix[i][j] = 0

    print("Failure Group Relationship Matrix:")
    print(relationship_matrix)
    print("\n")

    # Merge duplicated failure groups based on relationship matrix
    merged_groups = []
    visited = set()

    for i in range(n_groups):
        if i in visited:
            continue
        
        # Find all related groups for group i
        related_indices = [i]
        for j in range(n_groups):
            if i != j and relationship_matrix[i][j] == 1 and j not in visited:
                related_indices.append(j)
        
        # Merge all related groups
        merged_group = FailureGroup(
            id=len(merged_groups),
            error_msg=failure_group_list[i].error_msg,
            skipped=[],
            commands=[],
            tracees=[],
            duplicated=[],
            new_duplicated=[]
        )
        
        for idx in related_indices:
            visited.add(idx)
            fg = failure_group_list[idx]
            merged_group.skipped.extend(fg.skipped)
            merged_group.commands.extend(fg.commands)
            merged_group.tracees.extend(fg.tracees)
            merged_group.duplicated.extend(fg.duplicated)
            merged_group.new_duplicated.extend(fg.new_duplicated)
        
        # Remove duplicates
        merged_group.skipped = list(set(merged_group.skipped))
        merged_group.commands = list(set(merged_group.commands))
        merged_group.duplicated = list(set(merged_group.duplicated))
        merged_group.new_duplicated = list(set(merged_group.new_duplicated))
        
        merged_groups.append(merged_group)
    return merged_groups

def main():
    # Parse the UT failure list generated by check-ut.py 
    df = pd.read_csv("ut_failure_list.csv", delimiter='|', engine='python', header=None)
    df = df.rename(columns={df.columns[1]: 'Category', df.columns[2]: 'Class', df.columns[3]: 'Testcase', df.columns[4]: 'Result', df.columns[5]: 'ErrorMessage'})
    df = df[['Class', 'Testcase', 'Result', 'ErrorMessage']]

    
    import os
    issues_folder = os.path.curdir + "/issues"
    if os.path.exists(issues_folder) is True:
        import shutil
        shutil.rmtree(issues_folder)
    if os.path.exists(issues_folder) is False:
        os.mkdir(issues_folder)

    if args.merge:
        merged_issues_folder = os.path.curdir + "/merged_issues"
        if os.path.exists(merged_issues_folder) is True:
            import shutil
            shutil.rmtree(merged_issues_folder)
        if os.path.exists(merged_issues_folder) is False:
            os.mkdir(merged_issues_folder)

    if args.known:
        xpu_issues_folder = os.path.curdir + "/xpu_issue"
        if os.path.exists(xpu_issues_folder) is False:
            os.mkdir(xpu_issues_folder)
        # If check known issues download xpu-ops issues first
        download_all_open_issues_and_get_skiplist("intel/torch-xpu-ops", os.getenv("GITHUB_TOKEN"), xpu_issues_folder)

    # env_info = ""
    # import os
    # if os.path.exists(f'collect_env.py'):
    #     import subprocess
    #     env_info = subprocess.check_output(['python', 'collect_env.py']).decode()

    df_skip = None
    if args.skiplist is not None:
        df_skip = pd.read_csv("skiplist_map.csv", delimiter='|', engine='python', header=None)
        df_skip = df_skip.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df_skip = df_skip.rename(columns={df_skip.columns[0]: 'File', df_skip.columns[1]: 'Testcase', df_skip.columns[2]: 'Skipreason'})

    id = 0
    failure_group_list = []
    for group in df.groupby(['ErrorMessage']):
        failure_group = FailureGroup(id=id, error_msg=group[0][0], skipped=[], commands=[], tracees=[], duplicated=[], new_duplicated=[])
        for row in group[1].itertuples(index=False):
            if "::" in row.Class:
                test_class = row.Class.split("::")[-1].strip()
                test_file = row.Class.split('::')[0].strip()
                row_test_file = test_file.split('/')[-1]
                file_path = '/'.join(test_file.split('/')[:-1])
                test_class = test_file.replace('.py', '').replace('/', '.') + '.' + test_class
                file_path = 'test/' + file_path
                xml_file = file_path + '/' + test_file.replace('/', '.').replace('.py', '.xml')
            else:
                test_class = row.Class.strip()
                _test_file = '/'.join(test_class.split('.')[:-1])
                test_file = _test_file + '.py'
                xml_file = test_file.replace('.py', '.xml')
            test_case = row.Testcase.strip()
            line = f"op_ut,{test_class},{test_case}"
            if "_xpu.py" in test_file or "/xpu/" in test_file:
                if "extended" in test_file:
                    pytest_command = f"cd <pytorch>/third_party/torch-xpu-ops/test/xpu/extended "
                else:
                    pytest_command = f"cd <pytorch>/third_party/torch-xpu-ops/test/xpu "
            else:
                pytest_command = f"cd <pytorch>"
            pytest_command += f" && PYTORCH_TEST_WITH_SLOW=1 pytest -v {test_file} -k {test_case}"
            failure_group.skipped.append(line)
            failure_group.commands.append(pytest_command)
            
    
            if not os.path.exists(xml_file):
                if os.path.exists(f"third_party/torch-xpu-ops/test/xpu/op_ut_with_ext.{test_file.split('/')[-1].replace('.py', '.xml')}"):
                    xml_file = f"third_party/torch-xpu-ops/test/xpu/op_ut_with_ext.{test_file.split('/')[-1].replace('.py', '.xml')}"
                else:
                    if "::" in row.Class:
                        import glob
                        test_xml_files = glob.glob(os.path.join(file_path, f"{os.path.basename(xml_file).replace('.xml', '')}*"))
                        if len(test_xml_files) > 0:
                            for _xml_file in test_xml_files:
                                if os.path.exists(_xml_file):
                                    xml_file = _xml_file
                                    with open(_xml_file, 'r') as f:
                                        if test_case in f.read():
                                            xml_file = _xml_file
                                            break

            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for testcase in root.iter('testcase'):
                    if testcase.get('name') == test_case:
                        error = testcase.find('error')
                        failure = testcase.find('failure')
                        if error is not None:
                            failure_group.tracees.append(f"\n```\nCommand: {pytest_command}\n{error.text}\n```")
                        elif failure is not None:
                            failure_group.tracees.append(f"\n```\nCommand: {pytest_command}\n{failure.text}\n```")
                        else:
                            failure_group.tracees.append('')
                        break
            except Exception:
                failure_group.tracees.append('')

        # dump the basic issue information to a file group by error message
        write_issue_file(failure_group, group[0][0], issues_folder)

        failure_group_list.append(failure_group)

        # Create a matrix to map relationships between failure groups based on new_duplicated
        # Matrix values: 1 = similar issues (share duplicates), 0 = not similar
        id += 1

    if args.merge:
        for failure_group in failure_group_list:
            # Find duplicated issues for the issue group generated above
            failure_group.new_duplicated = get_duplicated_new_issues(failure_group, failure_group.error_msg, issues_folder, args.ratio)

        # Use merged groups if merge flag is enabled
        n_groups = len(failure_group_list)
        # Merge failure goups according to relationship matrix
        failure_group_list = merge_failure_groups(failure_group_list)
        print(f"Merged {n_groups} groups into {len(failure_group_list)} groups")
        # Write final issues
        for fg in failure_group_list:
            # Dump the merge issue groups
            issue_id = write_issue_file(fg, fg.error_msg, merged_issues_folder, submit=args.submit, skiplist=df_skip)
            if args.known:
                # Find duplicated issues of xpu-ops repo
                fg.duplicated = get_duplicated_known_issues(fg, fg.error_msg, xpu_issues_folder, args.ratio)
                if args.submit and len(fg.duplicated) > 0:
                    gh = Github_Issue("intel/torch-xpu-ops", os.getenv("GITHUB_TOKEN"))
                    duplicated_issues = [ f"#{_issue_id}" for _issue_id in fg.duplicated if _issue_id != issue_id ]
                    if duplicated_issues:
                        gh.add_comment(f"[xpu_triage_bot]:\n Possible related issues: {duplicated_issues}", issue_id)
                    else:
                        gh.add_comment(f"[xpu_triage_bot]:\n Possible related issues: Not found", issue_id)
                               

    
    print("\n\n### Final Failure Groups and their known duplicated issues:")
    print("issue_folder:", issues_folder if not args.merge else merged_issues_folder)

    if args.known:
        for fg in failure_group_list:
            print(f"Known duplicated issues for issue_group{fg.id}: {fg.duplicated}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GitHub issues from UT failure list")
    parser.add_argument('--merge', action='store_true', default=False, help="Merge similar issues")
    parser.add_argument('--known',action='store_true', default=False, help="Check known issues of torch-xpu-ops repo, set GITHUB_TOKEN with your token")
    parser.add_argument('--ai', action='store_true', default=False, help="Use AI for duplicate issue detection")
    parser.add_argument('--ratio', type=float, default=0.7, help="Similarity ratio threshold for duplicate detection")
    parser.add_argument('--submit', action='store_true', default=False, help="Submit issues to GitHub")
    parser.add_argument('--skiplist', type=str, help="skiplist")
    parser.add_argument('--stockci', action='store_true', default=False, help="Use stockci artifacts xml files")
    args = parser.parse_args()
    main()
