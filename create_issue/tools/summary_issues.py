import os
from get_duplicated_issues.get_duplicated_issues import download_all_open_issues_and_get_skiplist

with open("upstream_issues2.txt", "r") as f:
    issue_list = [f"{line.strip().split('/')[-1]}.txt" for line in f.readlines()]

xpu_issues_folder = os.path.curdir + "/xpu_issue"
if os.path.exists(xpu_issues_folder) is False:
    os.mkdir(xpu_issues_folder)

skip_list = download_all_open_issues_and_get_skiplist("intel/torch-xpu-ops", os.getenv("GITHUB_TOKEN"), xpu_issues_folder)
filtered_skip_list = [file for file in skip_list if file not in issue_list]
from get_duplicated_issues.get_duplicated_issues_ai import get_issues_summary
# get_issues_summary(issue_folder=xpu_issues_folder, issue_list=issue_list)
get_issues_summary(issue_folder=xpu_issues_folder, issue_list=filtered_skip_list)



