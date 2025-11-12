import pandas as pd

df = pd.read_csv("ut_failure_list.csv", delimiter='|', engine='python')
df = df.rename(columns={df.columns[1]: 'Category', df.columns[2]: 'Class', df.columns[3]: 'Testcase', df.columns[4]: 'Result', df.columns[5]: 'ErrorMessage'})
df = df[['Class', 'Testcase', 'Result', 'ErrorMessage']]
header="Cases:"
skipped = []
commands = []
traces = []
id = 0

env_info = ""
import os
if os.path.exists(f'collect_env.py'):
    import subprocess
    env_info = subprocess.check_output(['python', 'collect_env.py']).decode()

for group in df.groupby(['ErrorMessage']):
    for row in group[1].itertuples(index=False):
        test_class = row.Class.strip()
        _test_file = '/'.join(test_class.split('.')[:-1])
        test_file = _test_file + '.py'
        test_case = row.Testcase.strip()
        line = f"op_ut,{test_class},{test_case}"
        pytest_command = f"PYTORCH_TEST_WITH_SLOW=1 pytest -v {test_file} -k {test_case}"
        skipped.append(line)
        commands.append(pytest_command)
        xml_file = _test_file + '.xml'
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for testcase in root.iter('testcase'):
                if testcase.get('name') == test_case:
                    error = testcase.find('error')
                    failure = testcase.find('failure')
                    if error is not None:
                        traces.append(f"\n```\nCommand: {pytest_command}\n{error.text}```")
                    elif failure is not None:
                        traces.append(f"\n```\nCommand: {pytest_command}\n{failure.text}```")
                    else:
                        traces.append('')
                    break
        except Exception:
            traces.append('')

    with open(f'issues/issue_group{id}.txt', 'w') as f:
        f.write(f"Title: [Upstream] {group[0][0]}\n")

        cases = '\n'.join(skipped)
        f.write(f"Cases:\n{cases}\n")
        
        commands_str = '\n'.join(commands)
        f.write(f"\npytest_command:\n{commands_str}\n")
        
        f.write("\nTrace Example:\n")
        f.write(traces[-1])

        f.write(f"\n\nEnvironment Information:\n{env_info}\n")

    id += 1