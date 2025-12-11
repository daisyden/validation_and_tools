import os
import argparse
from get_duplicated_issues.github_issue import Github_Issue

parser = argparse.ArgumentParser(description="Create GitHub issues from UT failure list")
parser.add_argument('--file', type=str, help="issue file with tiltle in first line and body in the rest")
args = parser.parse_args()
gh = Github_Issue("intel/torch-xpu-ops", os.getenv("GITHUB_TOKEN"))
print(f"Submit merged issue file: {args.file}")
def create_github_issue(issuefile):
    with open(issuefile, 'r') as f:
        lines = f.readlines()   
        title = lines[0].strip()
        content = ''.join(lines[1:]).strip()
        issue_id = gh.create_issue_with_label(title, body=content, labels=["skipped", "port_from_skiplist"])
        print(f"issue_id is {issue_id}\n")

create_github_issue(args.file)
