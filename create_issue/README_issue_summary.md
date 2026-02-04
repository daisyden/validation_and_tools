Summary the github issues

# Command:
```
GITHUB_TOKEN=<token> bash issue_summary.sh <torch-xpu-ops nightly runid>
```

# What it do?
```
1. download nightly artifacts
2. run check-ut.py to collect failed ut and passed ut list
3. extract the unit test cases from github issue, collect error message and trace from nightly artifacts, add the latest status get from 2
4. dumnp the collected infomation to dynamic_skipped_list_xpuindex.csv
5. If dynamic_skipped_list_xpuindex.xlsx is provided with additional information, merge with dynamic_skipped_list_xpuindex.csv, and create dynamic_skipped_list_xpuindex_merged.xlsx 
```
