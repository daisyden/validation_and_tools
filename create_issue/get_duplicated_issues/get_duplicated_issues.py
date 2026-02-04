from .github_issue import Github_Issue 

def download_issue_content(issue):
    content = ""
    reporter = issue.user.login
    owner = issue.assignee.login if issue.assignee is not None else "Unassigned"
    title = issue.title
    skipped = "No"
    if "skipped" in [label.name for label in issue.labels]:
        skipped = "Yes"
    if issue.body is not None:
        content += issue.body + "\n"
    
    content = content.split('### Versions')[0]
    comments = issue.get_comments()
    comment = ""
    for _comment in comments:
        if _comment.body is not None:
            comment += _comment.body + "\n"
    return f"#{issue.id}\nReporter: {reporter}\nOwner: {owner}\nTitle: {title}\nSkipped: {skipped}\nBody: {content}\nComments: {comment}"

def download_all_open_issues_and_get_skiplist(repo:str, token:str, xpu_issues_folder:str):
    gh = Github_Issue(repo, token)
    issues = gh.get_issues(state="open")

    skip_list = []
    for issue in issues:
        if "skipped" in [label.name for label in issue.labels]:
            skip_list.append(f"{issue.number}.txt")
        import os
        if os.path.exists(f"{xpu_issues_folder}/{issue.number}.txt"):
            #print(f"Issue #{issue.number} already downloaded.")
            continue
        content = download_issue_content(issue)
        with open(f"{xpu_issues_folder}/{issue.number}.txt", "w", encoding="utf-8") as f:
            f.write(f"Issue #{issue.number}: {issue.title}\n")
            f.write(content)
            f.write("\n" + "="*80 + "\n\n")
            print(f"Downloaded issue #{issue.number} to file.")
    return skip_list

def download_all_open_issues_and_get_issue_with_label(repo:str, token:str, xpu_issues_folder:str, only_skipped:bool=False):
    gh = Github_Issue(repo, token)
    issues = gh.get_issues(state="open")

    issue_list = []
    for issue in issues:
        if only_skipped == False:
            issue_list.append((f"{issue.number}.txt", f"{issue.assignee.login if issue.assignee is not None else 'Unassigned'}", ','.join([label.name for label in issue.labels])))
        elif "skipped" in [label.name for label in issue.labels]:
            issue_list.append((f"{issue.number}.txt", f"{issue.assignee.login if issue.assignee is not None else 'Unassigned'}", ','.join([label.name for label in issue.labels])))

        import os
        if os.path.exists(f"{xpu_issues_folder}/{issue.number}.txt"):
            #print(f"Issue #{issue.number} already downloaded.")
            continue
        content = download_issue_content(issue)
        with open(f"{xpu_issues_folder}/{issue.number}.txt", "w", encoding="utf-8") as f:
            f.write(f"Issue #{issue.number}: {issue.title}\n")
            f.write(content)
            f.write("\n" + "="*80 + "\n\n")
            print(f"Downloaded issue #{issue.number} to file.")
        
    return issue_list

def get_duplicated_issues(id: str, skipped:list, error_message:str, trace:str, issue_folder:str, ratio: float):
    print(f"\n\n### Checking duplicated issues for group {id} with {error_message} ...\n")
    duplicated_issues = []
    import os, re

    def extract_errors_from_log(log_content):
        """
        Extract assertion errors and runtime errors from log content
        """
        # Patterns for different types of errors
        patterns = {
            'assertion_error': r'AssertionError:?(.*)',
            'runtime_error': r'RuntimeError:?(.*)',
            #'traceback': r'Traceback \(most recent call last\):\n(?:.*\n)*?(?:\w+Error:.*)',
            'any_python_error': r'^\w+Error:.*$',
            'exception': r'Exception:?(.*)',
            'value_error': r'ValueError:?(.*)',
            'type_error': r'TypeError:?(.*)',
            'index_error': r'IndexError:?(.*)',
            'key_error': r'KeyError:?(.*)',
            'import_error': r'ImportError:?(.*)',
            'crash': r'(.*)crash(.*)',
        }
        
        errors = {}
        
        for error_type, pattern in patterns.items():
            matches = re.findall(pattern, log_content, re.MULTILINE)
            if matches:
                errors[error_type] = matches
        
        return errors
    
    for issue_file in os.listdir(issue_folder):
        issue_file = os.path.join(issue_folder, issue_file)
        issue_file_id = issue_file.split('/')[-1].split('.')[0]
        print(f"## Checking issue file {issue_file_id}...")
        if issue_file.endswith(".txt") and issue_file_id != f"issue_group{id}":
            with open(issue_file, "r", encoding="utf-8") as f:
                content = f.read()
                # extract match with test_file, test_case and error message
                for skip in skipped:
                    if skip in content and "Skipped: Yes" in content:
                        print(f"# Skipping {skip} of issue_group{id} as it is marked skipped in {issue_file}")
                        duplicated_issues.append(issue_file_id)
                    else:
                        _test_file = '/'.join(skip.split(',')[1].split('.')[:-1])
                        test_file = _test_file + '.py'
                        test_case = skip.split(',')[2].strip()

                        if (f"{test_file}".replace('_xpu.py','.py').replace('test/', '') in content) and \
                            f"{test_case}" in content and \
                            f"{error_message}" in content:
                            duplicated_issues.append(issue_file_id)

                # Check whether error message is similar
                errors = extract_errors_from_log(content)
                print(f"Extracted errors {errors} from issue {issue_file_id}")

                from difflib import SequenceMatcher
                def similar(a, b):
                    return SequenceMatcher(None, a, b).ratio()

                for error_type in errors.keys():
                    if similar(error_message, f"{error_type}: {errors[error_type][0]}") > ratio:
                        print(f"\n# Found similar error message in issue {issue_file_id} with issue_group{id}: {error_type}: {errors[error_type][0]}    .vs    {error_message} \nsimilarity ratio is {similar(error_message, f'{error_type}: {errors[error_type][0]}')}")
                        duplicated_issues.append(issue_file_id)
                    else:
                        print(f"\n# No similar error message in issue {issue_file_id} with issue_group{id}: {error_type}: {errors[error_type][0]}    .vs    {error_message} similarity ratio is {similar(error_message, f'{error_type}: {errors[error_type][0]}')}")

    print(f"## Duplicated issues for group {id}: {duplicated_issues}")
    print("########################################\n\n")
    return duplicated_issues



