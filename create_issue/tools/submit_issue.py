import os
import argparse
from get_duplicated_issues.github_issue import Github_Issue

parser = argparse.ArgumentParser(description="Create GitHub issues from UT failure list")
parser.add_argument('--file', type=str, help="issue file with tiltle in first line and body in the rest")
parser.add_argument('--create', action='store_true', help="create issues on GitHub")

parser.add_argument('--comments', type=str, default="", help="comments file to map skip reasons")
parser.add_argument('--issuelist', type=str, default="", help="issue list file to avoid duplicated issues")
parser.add_argument('--labels', type=str, default="", help="labels to add to created issues, separated by commas")
parser.add_argument('--owner', type=str, default="", help="owner to assign to created issues")

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

if args.create and args.file:
    create_github_issue(args.file)


if args.issuelist is not None and len(args.issuelist) > 0:
    issues = args.issuelist.split(",")
    for issue in issues:
        issue = gh.get_issue(int(issue))
        if args.comments is not None and len(args.comments) > 0:
            gh.add_comment(args.comments, issue.number)
        if args.labels is not None and len(args.labels) > 0:
            labels = args.labels.split(",")
            for label in labels:
                gh.add_label(issue.number, label)
        if args.owner is not None and len(args.owner) > 0:
            gh.add_owner(issue.number, args.owner)