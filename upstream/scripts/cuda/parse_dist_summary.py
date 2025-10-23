import csv
import os
import sys

#!/usr/bin/env python3

def infer_testfile(category: str, ut: str) -> str:
    if category is None or ut is None:
        return ""
    category = category.strip().strip("/")
    ut = ut.strip()
    if not ut:
        return ""
    if ut.endswith(".py"):
        ut = ut[:-3]
    return f"test/{category}/{ut}.py" if category else f"{ut}.py"

def process(input_path: str, output_path: str):
    infile = sys.stdin if input_path == "-" else open(input_path, newline="", encoding="utf-8")
    with infile:
        reader = csv.reader(infile, delimiter='\t')
        rows_iter = iter(reader)
        try:
            header = next(rows_iter)
        except StopIteration:
            raise SystemExit("Empty CSV")
        # Locate Category and UT columns
        try:
            cat_idx = header.index(" Category ")
            ut_idx = header.index(" UT ")
        except ValueError as e:
            raise SystemExit("Required columns 'Category' and 'UT' not found") from e

        new_header = [" Testfile "] + header

        outfh = sys.stdout if output_path == "-" else open(output_path, "w", newline="", encoding="utf-8")
        with outfh:
            writer = csv.writer(outfh)
            writer.writerow(new_header)
            for row in rows_iter:
                # Ensure row has enough columns (pad if shorter)
                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))
                testfile = infer_testfile(row[cat_idx], row[ut_idx])
                writer.writerow([testfile] + row)

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python parse_dist_summary.py <input_csv|-] [output_csv|-]", file=sys.stderr)
        print("Defaults: output -> dist_summary_with_testfile.csv", file=sys.stderr)
        return
    input_path = args[0]
    output_path = args[1] if len(args) > 1 else "dist_summary_with_testfile.csv"
    process(input_path, output_path)

if __name__ == "__main__":
    main()
