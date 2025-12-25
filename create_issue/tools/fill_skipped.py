import os
from attr import dataclass
import pandas as pd
import argparse

def main():
    # Parse skip_list_common.py to map test cases to skip reasons
    skip_reason_map = {}
    skip_list_file = args.skiplist if args.skiplist else "skip_list_common.py"
    if os.path.exists(skip_list_file):
        with open(skip_list_file, 'r') as f:
            lines = f.readlines()
        
        current_test_file = None
        current_comment = []
        lastline_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                current_comment = []
                continue
            
            # Capture comments as potential skip reasons
            if stripped.startswith('#'):
                if lastline_comment:
                    current_comment.append(stripped[1:].strip())
                else:
                    current_comment = [stripped[1:].strip()]
                lastline_comment = True
                continue
            else:
                lastline_comment = False
            
            # Match test file keys (e.g., "test/path/to/file.py": [)
            if '":' in stripped and not stripped.startswith('#'):
                # Extract test file name
                parts = stripped.split('"')
                if len(parts) >= 2:
                    current_test_file = parts[1]
                    current_comment = []
                continue
            
            # Match test case entries
            if current_test_file and '"' in stripped:
                # Extract test case name
                test_case = stripped.split('"')[1] if '"' in stripped else None
                
                if test_case:
                    # Check for inline comment (reason after the test case)
                    inline_comment = None
                    if '#' in stripped:
                        comment_parts = stripped.split('#', 1)
                        if len(comment_parts) > 1:
                            inline_comment = comment_parts[1].strip().rstrip(',')
                    
                    # Use inline comment if available, otherwise use the preceding comment
                    reason = inline_comment if inline_comment else ("; ".join(current_comment) if current_comment else "No reason provided")
                    
                    # Create key as "test_file::test_case"
                    key = f"{current_test_file}|{test_case}"
                    skip_reason_map[key] = reason
                    
                    # # Reset current_comment after using it
                    # if not inline_comment:
                    #     current_comment = None

        print(f"Loaded {len(skip_reason_map)} skip reasons from {skip_list_file}: \n ")
        for k, v in skip_reason_map.items():
            print(f"{k} | {v}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GitHub issues from UT failure list")
    parser.add_argument('--skiplist', type=str, default=False, help="skip list")
    args = parser.parse_args()
    main()
