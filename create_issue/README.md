Example:
run pytest with --junit-xml=test_ops_gradients.xml
```
mkdir test
mv test_ops_graidents.xml test/.
python check-ut.py test/test_ops_gradients.xml
```
ut_failure_list.csv will be generated

Command:
```
# merge issues and check duplicated issue with xpu-ops repo, ratio is the threashold to check similarity
GITHUB_TOKEN=<token> python create_issue.py --merge True --known True --ratio 0.7

# merge issues
GITHUB_TOKEN=<token> python create_issue.py --merge True --ratio 0.7

# only generate issue according to error message
GITHUB_TOKEN=<token> python create_issue.py 

```



Result example with merge and known issue similarity check:
```
### Final Failure Groups and their known duplicated issues:
issue_folder: ./merged_issues
Known duplicated issues for issue_group0: ['2358', '2270']
Known duplicated issues for issue_group1: ['2249']
Known duplicated issues for issue_group2: []
Known duplicated issues for issue_group3: []
```
That means the result is put under merged_issues folder and the issue_group0 could be duplicated with #2358 and #2270. issue_group1 could be duplicated with #2249


