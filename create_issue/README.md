Example:
run pytest with --junit-xml=test_ops_gradients.xml
```
mkdir test
# The xml file folder could be put under the same directory as test case, for example torch-xpu-ops test, put xml under third_party/torch-xpu-ops/test/xpu
mv test_ops_graidents.xml test/.   
python check-ut.py test/test_ops_gradients.xml
```
ut_failure_list.csv will be generated

Command:
```
# parse the skip_list_common.py and generate a csv file with skip reason mapped to test case.
# skiplist_map.csv will be created
python fill_skipped.py

# merge issues and check duplicated issue with xpu-ops repo, ratio is the threashold to check similarity and it is adjsutable
# finially the merged issue will be submitted to github. Please ensure the "repo" scope is checked for your github token.
GITHUB_TOKEN=<token> python create_issue.py --merge --known --ratio 0.7 --submit

# merge issues and check duplicated issue with xpu-ops repo, ratio is the threashold to check similarity and it is adjsutable
# finially the merged issue will be submitted to github, the skiplist comments would also be added in description. Please ensure the "repo" scope is checked for your github token.
GITHUB_TOKEN=<token> python create_issue.py --merge --known --ratio 0.7 --submit --skiplist skiplistt_map.csv


# merge issues
GITHUB_TOKEN=<token> python create_issue.py --merge --ratio 0.7

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


