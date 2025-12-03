#!/usr/bin/env python3
"""
Extract test details from JUnit XML files and save to Excel/CSV.

Usage:
    python get_details.py --input "results/*.xml" --output details.xlsx
    python get_details.py --input file1.xml file2.xml --output details.csv
    python get_details.py --input results_dir --output test_details.xlsx
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Set, Any
import argparse
import re
import logging
import glob
from contextlib import contextmanager
from functools import lru_cache
import time

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TestCase:
    """Immutable data class representing a single test case."""
    device: str
    testtype: str
    testfile: str
    uniqname: str
    classname: str
    name: str
    status: str
    time: float
    source_file: str
    message: str = ""
    type: str = ""


class TestDetailsExtractor:
    """
    Extracts test details from JUnit XML files.
    """

    # Test type mappings based on file patterns
    TEST_TYPE_MAPPINGS = {
        'cuda-all': ['nvidia.gpu', 'dgx.b200'],
        'xpu-dist': ['xpu_distributed'],
        'xpu-ops': ['op_ut_with_'],
        'xpu-stock': ['/test-reports/'],
    }

    # Status standardization mappings
    STATUS_MAPPINGS = {
        'passed': ['pass', 'success'],
        'xfail': ['xfail'],
        'failed': ['fail', 'error'],
        'skipped': ['skip']
    }

    def __init__(self):
        """Initialize the test details extractor."""
        self.all_test_cases: List[TestCase] = []
        self.unique_total_case: int = 0
        self.processed_files: List[str] = []
        self.empty_case_files: List[str] = []
        
        # Compile regex patterns for better performance
        self._classname_pattern = re.compile(r'.*\.')
        self._casename_pattern = re.compile(r'[^a-zA-Z0-9_.-]')
        self._testfile_pattern = re.compile(r'.*torch-xpu-ops\.test\.xpu\.')
        self._normalize_pattern = re.compile(r'.*\.\./test/')
        self._gpu_pattern = re.compile(r'(?:xpu)', re.IGNORECASE)

    @contextmanager
    def _timer(self, operation: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"{operation} completed in {elapsed:.3f}s")

    def find_xml_files(self, input_paths: List[str]) -> List[Path]:
        """
        Find all XML files from various input specifications.
        """
        xml_files: Set[Path] = set()

        for input_path in input_paths:
            path = Path(input_path)

            if path.is_file() and path.suffix.lower() == '.xml':
                xml_files.add(path.resolve())
            elif path.is_dir():
                xml_files.update(path.resolve().glob('**/*.xml'))
            else:
                # Handle glob patterns
                for file_path in glob.glob(input_path, recursive=True):
                    file_path = Path(file_path)
                    if file_path.is_file() and file_path.suffix.lower() == '.xml':
                        xml_files.add(file_path.resolve())

        return sorted(xml_files)

    def parse_single_xml(self, xml_file: Path) -> Tuple[Optional[ET.ElementTree], Optional[ET.Element]]:
        """
        Parse a single XML file with error handling.
        """
        try:
            if not xml_file.exists():
                logger.error(f"XML file not found: {xml_file}")
                return None, None

            with self._timer(f"Parsing {xml_file.name}"):
                tree = ET.parse(xml_file)
                root = tree.getroot()
            return tree, root

        except ET.ParseError as e:
            logger.error(f"Error parsing XML file {xml_file}: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error parsing {xml_file}: {e}")
            return None, None

    @lru_cache(maxsize=128)
    def _determine_test_type(self, xml_file: Path) -> str:
        """
        Determine test type based on XML file path patterns.
        """
        xml_file_str = str(xml_file)
        return next(
            (test_type for test_type, patterns in self.TEST_TYPE_MAPPINGS.items() 
             if any(pattern in xml_file_str for pattern in patterns)),
            'xpu-xpu'
        )

    def _extract_testfile(self, classname: str, filename: str, xml_file: Path) -> str:
        """
        Extract test file path from available information.
        """
        # Priority 1: Use filename from XML
        if filename and filename.endswith('.py'):
            testfile = f'test/{filename}'
        # Priority 2: Extract from classname
        elif classname:
            testfile = self._testfile_pattern.sub('test/', classname).replace('.', '/')
            if '/' in testfile:
                testfile = testfile.rsplit('/', 1)[0] + '.py'
            else:
                testfile = f'{testfile}.py'
        # Priority 3: Extract from XML filename
        else:
            xml_file_str = str(xml_file)
            testfile = (
                re.sub(r'.*op_ut_with_[a-zA-Z0-9]+\.', 'test.', xml_file_str)
                .replace('.', '/')
                .replace('/py/xml', '.py')
                .replace('/xml', '.py')
            )
        # Replace specific strings
        output = self._normalize_pattern.sub('test/', testfile).replace(
                                'test/test/', 'test/'
                            ).replace(
                                "_xpu.py", ".py"
                            ).replace(
                                "test_c10d_xccl.py", "test_c10d_nccl.py"
                            ).replace(
                                "test_c10d_ops_xccl.py", "test_c10d_ops_nccl.py"
                            )
        return output

    def _extract_classname(self, full_classname: str) -> str:
        """Extract simplified classname from full classname."""
        if not full_classname:
            return "UnknownClass"
        return self._classname_pattern.sub('', full_classname)

    def _extract_casename(self, casename: str) -> str:
        """Extract normalized test case name."""
        if not casename:
            return "unknown_name"
        try:
            casename = self._casename_pattern.sub('', casename)
            casename = self._testfile_pattern.sub('', casename)
            return casename or "error_name"
        except Exception:
            return "error_name"

    def _generate_uniqname(self, filename: str, classname: str, name: str) -> str:
        """Generate unique identifier for test case."""
        combined = f"{filename}{classname}{name}"
        return self._gpu_pattern.sub('cuda', combined)

    def _determine_test_status(self, testcase: ET.Element) -> Tuple[str, str, str]:
        """
        Determine test status and extract message/type.
        """
        # Check for failure
        failure = testcase.find('failure')
        if failure is not None:
            message = failure.get('message', '')
            return ('xfail', message, 'xfail') if 'pytest.xfail' in message else ('failure', message, 'failure')

        # Check for skip
        skipped = testcase.find('skipped')
        if skipped is not None:
            message = skipped.get('message', '')
            skip_type = skipped.get('type', '')
            if 'pytest.xfail' in skip_type or 'pytest.xfail' in message:
                return 'xfail', message, 'xfail'
            return 'skipped', message, skip_type

        return 'passed', '', ''

    def _process_testcase_element(self, testcase: ET.Element, xml_file: Path, test_type: str) -> Optional[TestCase]:
        """
        Process a single testcase element into a TestCase object.
        """
        try:
            classname = testcase.get('classname', 'UnknownClass')
            filename = testcase.get('file', 'unknown_file')
            name = testcase.get('name', 'unknown_name')
            time = float(testcase.get('time', 0))

            # Normalize extracted data
            simplified_classname = self._extract_classname(classname)
            simplified_casename = self._extract_casename(name)
            testfile = self._extract_testfile(classname, filename, xml_file)
            uniqname = self._generate_uniqname(testfile, simplified_classname, simplified_casename)

            status, message, result_type = self._determine_test_status(testcase)

            return TestCase(
                device=test_type.split('-')[0],
                testtype=test_type,
                testfile=testfile,
                uniqname=uniqname,
                classname=simplified_classname,
                name=simplified_casename,
                status=status,
                time=time,
                source_file=str(xml_file),
                message=message,
                type=result_type
            )

        except Exception as e:
            logger.error(f"Error processing test case: {e}")
            return None

    def process_single_xml_file(self, xml_file: Path) -> bool:
        """
        Process a single XML file and extract test results.
        """
        tree, root = self.parse_single_xml(xml_file)
        if not root:
            return False

        test_type = self._determine_test_type(xml_file)
        file_test_cases = []

        # Extract individual test cases
        test_cases_elements = root.findall('.//testcase')
        
        if test_cases_elements:
            for testcase in tqdm(test_cases_elements, 
                               desc=f"Processing {xml_file.name}", 
                               leave=False,
                               unit="test"):
                test_case = self._process_testcase_element(testcase, xml_file, test_type)
                if test_case:
                    file_test_cases.append(test_case)
        else:
            self.empty_case_files.append(str(xml_file))

        # Add to consolidated results
        self.all_test_cases.extend(file_test_cases)
        self.processed_files.append(str(xml_file))

        logger.debug(f"Processed {len(file_test_cases)} test cases from {xml_file.name}")
        return True

    def process_multiple_xml_files(self, xml_files: List[Path]) -> bool:
        """
        Process multiple XML files with progress bar.
        """
        logger.info(f"Starting processing of {len(xml_files)} XML files")
        no_case_files_list = []

        success_count = 0
        for xml_file in tqdm(xml_files, desc="Processing XML files", unit="file"):
            if self.process_single_xml_file(xml_file):
                success_count += 1

        logger.info(f"Successfully processed {success_count}/{len(xml_files)} files")
        return success_count > 0

    def standardize_status_value(self, status: Any) -> str:
        """Standardize status values for consistent comparison."""
        if pd.isna(status):
            return str(status)
        
        status_str = str(status).lower()
        
        for standardized_status, patterns in self.STATUS_MAPPINGS.items():
            if any(pattern in status_str for pattern in patterns):
                return standardized_status
        
        return status_str

    def drop_deduplicated(self, df: pd.DataFrame, group_cols: list, status_col: str) -> pd.DataFrame:
        """Optimized function for deduplication with status priority"""
        # Define priority mapping
        priority_map = {'pass': 5, 'xfail': 4, 'fail': 3, 'skip': 2, '': 1}
        
        # Add priority and get index of min priority per group
        df = df.copy()

        df['_priority'] = df[status_col].map(priority_map).fillna(-1)
        
        # Find indices of best rows
        best_indices = df.groupby(group_cols)['_priority'].idxmax()
        
        # Return best rows
        result = df.loc[best_indices].drop('_priority', axis=1).reset_index(drop=True)
        
        return result

    def safe_merge(self, left: pd.DataFrame, right: pd.DataFrame, merge_keys: List[str]) -> pd.DataFrame:
        """
        Safely merge two DataFrames with proper suffix handling.

        Args:
            left: Left DataFrame
            right: Right DataFrame
            merge_keys: Keys to merge on

        Returns:
            Merged DataFrame
        """
        # Identify overlapping columns (excluding merge keys)
        left_cols = set(left.columns)
        right_cols = set(right.columns)
        overlapping_cols = (left_cols & right_cols) - set(merge_keys)

        if overlapping_cols:
            logger.debug(f"Overlapping columns found: {overlapping_cols}")
            # Use suffixes to distinguish overlapping columns
            suffixes = ('_cuda', '_xpu')
        else:
            suffixes = ('', '')

        try:
            merged_df = pd.merge(left, right, on=merge_keys, how='outer', suffixes=suffixes)
            logger.debug(f"Merge successful: {len(left)} + {len(right)} -> {len(merged_df)} rows")
            return merged_df
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            logger.error(f"Left columns: {list(left.columns)}")
            logger.error(f"Right columns: {list(right.columns)}")
            raise

    def save_test_details(self, output_file: str) -> None:
        """
        Save all test details to Excel or CSV file.
        """
        if not self.all_test_cases:
            logger.warning("No test cases to export")
            return

        try:
            df = pd.DataFrame([asdict(tc) for tc in self.all_test_cases])
            
            # Select and order columns for better readability
            column_order = [
                'device', 'testtype', 'uniqname', 'testfile', 'classname', 'name', 
                'status', 'message', 'type', 'time'
            ]
            available_columns = [col for col in column_order if col in df.columns]
            df = df[available_columns]
            df['status'] = df['status'].apply(self.standardize_status_value)
            # Remove duplicated cases
            df = df.sort_values(['status', 'time'], ascending=[True, False])
            grouped_df = self.drop_deduplicated(df, ['device', 'uniqname', 'classname', 'name'], 'status')
            self.unique_total_case = len(grouped_df)

            # Save to file
            output_path = Path(output_file)
            if output_path.suffix.lower() in ['.xlsx', '.xls']:
                grouped_df.to_excel(output_file, index=False, engine='openpyxl')
            else:
                grouped_df.to_csv(output_file, index=False)
            # Merged data
            cuda_df = grouped_df.loc[(grouped_df['device'] == 'cuda')]
            xpu_df = grouped_df.loc[(grouped_df['device'] == 'xpu')]
            MERGE_KEYS = ['uniqname']
            merged_df = self.safe_merge(cuda_df, xpu_df, MERGE_KEYS)
            if output_path.suffix.lower() in ['.xlsx', '.xls']:
                merged_df.to_excel(output_file.replace('.xls', '_merged.xls'), index=False, engine='openpyxl')
            else:
                merged_df.to_csv(output_file.replace('.csv', '_merged.csv'), index=False)
            # Skipped data
            skipped_df = merged_df.loc[(merged_df['status_cuda'] == 'passed') & (merged_df['status_xpu'] != 'passed')]
            if output_path.suffix.lower() in ['.xlsx', '.xls']:
                skipped_df.to_excel(output_file.replace('.xls', '_xpu_only_skipped.xls'), index=False, engine='openpyxl')
            else:
                skipped_df.to_csv(output_file.replace('.csv', '_xpu_only_skipped.csv'), index=False)

            logger.info(f"Saved {len(grouped_df)} test details to: {output_file}")

        except Exception as e:
            logger.error(f"Error saving test details: {e}")
            raise

    def extract_details(self, input_paths: List[str], output_file: str) -> bool:
        """
        Main method to extract details from XML files.
        """
        logger.info(f"Starting details extraction from {len(input_paths)} input paths")

        # Find XML files
        xml_files = self.find_xml_files(input_paths)

        if not xml_files:
            logger.error("No XML files found matching the input criteria")
            return False

        # Process files
        if not self.process_multiple_xml_files(xml_files):
            logger.error("No files were successfully processed")
            return False

        # Save results
        with self._timer(f"Saving to {output_file}"):
            self.save_test_details(output_file)
        
        logger.info(f"Details extraction completed successfully. Processed {len(self.all_test_cases)} test cases.")
        return True


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Extract test details from JUnit XML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process multiple specific files
  python get_details.py --input results1.xml results2.xml results3.xml --output details.xlsx

  # Process all XML files in a directory
  python get_details.py --input results/ --output details.csv

  # Process files using glob pattern
  python get_details.py --input "results/*.xml" --output test_details.xlsx
        """
    )

    parser.add_argument(
        '--input', '-i',
        nargs='+',
        required=True,
        help='XML file paths, directories, or glob patterns'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file path (supports .xlsx, .xls, .csv)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Initialize and run extractor
    extractor = TestDetailsExtractor()
    
    if not extractor.extract_details(args.input, args.output):
        return 1

    print(f"\n✓ Details extraction complete!")
    print(f"  - Output file: {args.output}")
    print(f"  - Processed files: {len(extractor.processed_files)}")
    print(f"  - Empty case files: {len(extractor.empty_case_files)}")
    print(f"  - Total test cases: {len(extractor.all_test_cases)}")
    print(f"  - Unique test cases: {extractor.unique_total_case}")

    return 0


if __name__ == "__main__":
    exit(main())
