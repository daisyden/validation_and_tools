#!/usr/bin/env python3
"""
CSV Test Result Merger

Merge multiple test result CSV files from different workflows (CUDA, XPU, XPU-Ops, Distributed)
into a unified view for comparison and analysis.

Usage:
    python merge_test_results.py --f1 cuda_results.csv --f2 xpu_results.csv
           --f3 xpu_ops_results.csv --f4 distributed_results.csv -o merged_results.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResultMerger:
    """Merges test results from multiple CSV files for comparison analysis."""

    # Column name mappings for different input files
    COLUMN_MAPPINGS = {
        'f1': {  # Reference (CUDA)
            0: 'Testfile', 1: 'Class_unified', 2: 'Testcase_unified',
            3: 'Class', 4: 'Testcase',
            5: 'Result', 6: 'SkipReason'
        },
        'f2': {  # Stock XPU
            0: 'Testfile', 1: 'Class_unified', 2: 'Testcase_unified',
            3: 'Class-stock-xpu', 4: 'Testcase-stock-xpu',
            5: 'Result-stock-xpu', 6: 'SkipReason-stock-xpu'
        },
        'f3': {  # XPU-Ops
            0: 'Testfile', 1: 'Class_unified', 2: 'Testcase_unified',
            3: 'Class-xpu-ops', 4: 'Testcase-xpu-ops',
            5: 'Result-xpu-ops', 6: 'SkipReason-xpu-ops'
        },
        'f4': {  # Distributed XPU-Ops
            0: 'Testfile', 1: 'Class_unified', 2: 'Testcase_unified',
            3: 'Class-xpu-dist', 4: 'Testcase-xpu-dist',
            5: 'Result-xpu-dist', 6: 'SkipReason-xpu-dist'
        }
    }

    # Merge keys
    MERGE_KEYS = ['Testfile', 'Class_unified', 'Testcase_unified']

    def __init__(self):
        """Initialize the test result merger."""
        self.processed_files = {}

    def load_and_clean_dataframe(self, file_path: Optional[str]) -> Optional[pd.DataFrame]:
        """
        Load CSV file and perform initial cleaning.

        Args:
            file_path: Path to CSV file, or None if file doesn't exist

        Returns:
            Cleaned DataFrame or None if file doesn't exist

        Raises:
            FileNotFoundError: If file_path is provided but doesn't exist
        """
        if file_path is None:
            return None

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Loading file: {path}")

        # Handle potential encoding issues
        try:
            df = pd.read_csv(path, delimiter='|')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {path}, trying latin-1")
            df = pd.read_csv(path, delimiter='|', encoding='latin-1')

        # Remove "**Total**" row if present
        if len(df) > 0 and df.iloc[-1, 1] == "**Total**":
            df = df.iloc[:-1]
            logger.debug("Removed '**Total**' summary row")

        # Clean up any unnamed columns to prevent merge conflicts
        df = self._clean_unnamed_columns(df)

        return df

    def _clean_unnamed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove or rename unnamed columns to avoid merge conflicts.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned column names
        """
        # Identify unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]

        if unnamed_cols:
            logger.debug(f"Found unnamed columns: {unnamed_cols}")

            # Remove unnamed columns if they're empty
            for col in unnamed_cols:
                if df[col].isna().all() or (df[col] == '').all():
                    df = df.drop(columns=[col])
                    logger.debug(f"Dropped empty unnamed column: {col}")
                else:
                    # Rename unnamed columns with meaningful names
                    col_idx = df.columns.get_loc(col)
                    new_name = f"Extra_Column_{col_idx}"
                    df = df.rename(columns={col: new_name})
                    logger.debug(f"Renamed unnamed column {col} to {new_name}")

        return df

    def rename_columns(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """
        Rename DataFrame columns according to predefined mappings.

        Args:
            df: Input DataFrame
            file_type: Type of file ('f1', 'f2', 'f3', 'f4')

        Returns:
            DataFrame with renamed columns
        """
        if file_type not in self.COLUMN_MAPPINGS:
            raise ValueError(f"Unknown file type: {file_type}")

        column_mapping = self.COLUMN_MAPPINGS[file_type]

        # Only rename columns that exist in the DataFrame
        rename_dict = {}
        for i, new_name in column_mapping.items():
            if i < len(df.columns):
                rename_dict[df.columns[i]] = new_name
            else:
                logger.warning(f"Column index {i} not found in {file_type}, has {len(df.columns)} columns")

        # Handle any extra columns beyond our mapping
        extra_cols = [col for col in df.columns if col not in rename_dict]
        for i, col in enumerate(extra_cols):
            rename_dict[col] = f"{file_type}_extra_{i}"
            logger.debug(f"Renaming extra column {col} to {rename_dict[col]}")

        return df.rename(columns=rename_dict)

    def filter_dataframe(self, df: pd.DataFrame, include_pattern: str, exclude_pattern: str) -> pd.DataFrame:
        """
        Filter DataFrame based on include/exclude patterns.

        Args:
            df: DataFrame to filter
            include_pattern: Pattern to include (substring match)
            exclude_pattern: Pattern to exclude (substring match)

        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()

        # Apply include filter
        if include_pattern and len(include_pattern) > 0:
            mask = filtered_df.iloc[:, 0].str.contains(include_pattern, na=False)
            filtered_df = filtered_df[mask]
            logger.info(f"Applied include filter '{include_pattern}': {len(filtered_df)} rows remaining")

        # Apply exclude filter
        if exclude_pattern and len(exclude_pattern) > 0:
            mask = ~filtered_df.iloc[:, 0].str.contains(exclude_pattern, na=False)
            filtered_df = filtered_df[mask]
            logger.info(f"Applied exclude filter '{exclude_pattern}': {len(filtered_df)} rows remaining")

        return filtered_df

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
            suffixes = ('_left', '_right')
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

    def merge_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames using outer join.

        Args:
            dataframes: Dictionary of DataFrames to merge

        Returns:
            Merged DataFrame
        """
        if not dataframes:
            raise ValueError("No DataFrames to merge")

        # Start with first DataFrame
        merged_df = None
        dataframes_list = list(dataframes.items())

        for i, (name, df) in enumerate(dataframes_list):
            logger.info(f"Processing {name}: {len(df)} rows, columns: {list(df.columns)}")

            if merged_df is None:
                merged_df = df
                logger.info(f"Starting merge with {name}")
            else:
                before_count = len(merged_df)
                merged_df = self.safe_merge(merged_df, df, self.MERGE_KEYS)
                after_count = len(merged_df)
                logger.info(f"Merged {name}: {before_count} -> {after_count} rows")

        return merged_df

    def calculate_xpu_result(self, row: pd.Series) -> str:
        """
        Calculate overall XPU result based on individual XPU implementations.

        Logic:
        - If any XPU implementation passed -> 'passed'
        - If any XPU implementation failed -> 'failed'
        - If any XPU implementation xfailed -> 'xfail'
        - If all XPU implementations skipped or empty -> 'skipped'

        Args:
            row: DataFrame row containing XPU result columns

        Returns:
            Overall XPU result string
        """
        xpu_columns = ['Result-stock-xpu', 'Result-xpu-ops', 'Result-xpu-dist']

        # Get results from available XPU columns
        results = []
        for col in xpu_columns:
            if col in row and pd.notna(row[col]) and row[col] != '':
                results.append(str(row[col]).lower().strip())

        if not results:
            return ''

        # Check for passed results first
        if any(result == 'passed' for result in results):
            return 'passed'

        # Check for failed results
        if any(result == 'failed' for result in results):
            return 'failed'

        # Check for xfail results
        if any(result in ['xfail', 'xfailed'] for result in results):
            return 'xfail'

        # Check if all are skipped
        if all(result == 'skipped' for result in results):
            return 'skipped'

        # Default case
        return ''

    def calculate_xpu_result_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized version of calculate_xpu_result for better performance.

        Args:
            df: DataFrame containing XPU result columns

        Returns:
            Series with calculated XPU results
        """
        xpu_columns = ['Result-stock-xpu', 'Result-xpu-ops', 'Result-xpu-dist']

        # Create a copy of relevant columns and fill NaN
        result_data = df[xpu_columns].copy()
        for col in xpu_columns:
            result_data[col] = result_data[col].fillna('').str.lower().str.strip()

        # Create conditions using vectorized operations
        has_passed = (result_data == 'passed').any(axis=1)
        has_failed = (result_data == 'failed').any(axis=1)
        has_xfail = (result_data.isin(['xfail', 'xfailed'])).any(axis=1)
        all_skipped = (result_data == 'skipped').all(axis=1)
        all_empty = (result_data == '').all(axis=1)

        # Apply conditions in priority order
        result = pd.Series('', index=df.index)
        result[all_skipped] = 'skipped'
        result[all_empty] = ''
        result[has_xfail] = 'xfail'
        result[has_failed] = 'failed'
        result[has_passed] = 'passed'

        return result

    def generate_analysis_reports(self, merged_df: pd.DataFrame, output_base: str) -> None:
        """
        Generate analysis reports from merged data.

        Args:
            merged_df: Merged DataFrame containing all test results
            output_base: Base path for output files
        """
        output_path = Path(output_base)

        # Check which XPU result columns actually exist
        available_xpu_cols = []
        xpu_columns_to_check = ['Result-stock-xpu', 'Result-xpu-ops', 'Result-xpu-dist']

        for col in xpu_columns_to_check:
            if col in merged_df.columns:
                available_xpu_cols.append(col)

        if not available_xpu_cols:
            logger.warning("No XPU result columns found for analysis")
            return

        # Report 1: All XPU skipped tests
        xpu_conditions = []
        for col in available_xpu_cols:
            xpu_conditions.append((merged_df[col] != 'passed') | (merged_df[col].isna()))

        # Combine conditions with AND (all XPU implementations failed/skipped)
        combined_condition = xpu_conditions[0]
        for condition in xpu_conditions[1:]:
            combined_condition = combined_condition & condition

        xpu_all_skipped = merged_df[combined_condition]

        xpu_all_skipped_path = output_path.parent / "xpu_all_skipped.csv"
        xpu_all_skipped.to_csv(xpu_all_skipped_path, index=False)
        logger.info(f"XPU all skipped report: {len(xpu_all_skipped)} tests -> {xpu_all_skipped_path}")

        # Report 2: XPU only skipped tests (passed on CUDA but failed/skipped on XPU)
        if 'Result' in merged_df.columns:  # CUDA results
            cuda_passed = merged_df['Result'] == 'passed'
            xpu_only_skipped = merged_df[cuda_passed & combined_condition]

            xpu_only_skipped_path = output_path.parent / "xpu_only_skipped.csv"
            xpu_only_skipped.to_csv(xpu_only_skipped_path, index=False)
            logger.info(f"XPU only skipped report: {len(xpu_only_skipped)} tests -> {xpu_only_skipped_path}")

            # Generate command file for skipped tests
            self.generate_test_commands(xpu_only_skipped, output_path.parent / "xpu_skipped_commands.txt")

    def generate_test_commands(self, df: pd.DataFrame, output_file: Path) -> None:
        """
        Generate pytest commands for skipped tests.

        Args:
            df: DataFrame containing skipped tests
            output_file: Path to output command file
        """
        commands = []
        for _, row in df.iterrows():
            testfile = row['Testfile']
            # Use CUDA testcase name and replace with XPU
            testcase = row.get('Testcase', '')
            if testcase:
                testcase = testcase.replace("cuda", "xpu")
                command = f"pytest -v {testfile} -k {testcase}"
                commands.append(command)

        with open(output_file, 'w') as f:
            f.write("\n".join(commands))

        logger.info(f"Generated {len(commands)} test commands -> {output_file}")

    def split_large_output(self, df: pd.DataFrame, output_path: Path, num_parts: int = 4) -> None:
        """
        Split large DataFrame into multiple files for easier handling.

        Args:
            df: DataFrame to split
            output_path: Base output path
            num_parts: Number of parts to split into
        """
        base_name = output_path.stem
        parent_dir = output_path.parent

        chunks = np.array_split(df, num_parts)

        logger.info(f"Splitting output into {num_parts} parts...")
        for i, chunk in enumerate(tqdm(chunks, desc="Writing split files")):
            part_path = parent_dir / f"{base_name}_part_{i+1:03d}.csv"
            chunk.to_csv(part_path, index=False)

        logger.info(f"Created {num_parts} split files in {parent_dir}")

    def merge(
        self,
        f1: str,
        f2: str,
        f3: Optional[str],
        f4: Optional[str],
        output: str,
        include_pattern: Optional[str],
        exclude_pattern: Optional[str],
        split_output: bool = False
    ) -> None:
        """
        Main merge function coordinating the entire workflow.

        Args:
            f1: Path to first CSV file (reference/CUDA)
            f2: Path to second CSV file (stock XPU)
            f3: Path to third CSV file (XPU-Ops)
            f4: Path to fourth CSV file (distributed XPU-Ops)
            output: Output file path
            include_pattern: Pattern to include in filtering
            exclude_pattern: Pattern to exclude in filtering
            split_output: Whether to split output into multiple files
        """
        logger.info("Starting test result merge process")

        # Load all dataframes
        dataframes = {}
        for i, (name, path) in enumerate([
            ('f1', f1), ('f2', f2), ('f3', f3), ('f4', f4)
        ], 1):
            try:
                df = self.load_and_clean_dataframe(path)
                if df is not None:
                    df = self.rename_columns(df, name)
                    dataframes[name] = df
                    logger.info(f"Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"Failed to load {name} ({path}): {e}")

        if not dataframes:
            raise ValueError("No valid input files provided")

        # Merge dataframes
        merged_df = self.merge_dataframes(dataframes)
        logger.info(f"Initial merge complete: {len(merged_df)} total rows, {len(merged_df.columns)} columns")

        # Calculate XPU result using vectorized approach for better performance
        if len(merged_df) > 1000:
            logger.info("Using vectorized XPU result calculation for large dataset")
            merged_df['Result-merge-xpu'] = self.calculate_xpu_result_vectorized(merged_df)
        else:
            logger.info("Using row-wise XPU result calculation")
            merged_df['Result-merge-xpu'] = merged_df.apply(self.calculate_xpu_result, axis=1)

        # Apply filters
        if include_pattern or exclude_pattern:
            merged_df = self.filter_dataframe(merged_df, include_pattern, exclude_pattern)
            logger.info(f"After filtering: {len(merged_df)} rows")

        # Save main output
        output_path = Path(output)
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Main output saved: {output_path}")

        # Generate analysis reports
        self.generate_analysis_reports(merged_df, output)

        # Split output if requested
        if split_output and len(merged_df) > 1000:  # Only split if large dataset
            self.split_large_output(merged_df, output_path)

        logger.info("Merge process completed successfully!")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Merge test result CSV files from different workflows',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input files
    parser.add_argument(
        '--f1',
        default='data/cases/details_cuda.csv',
        help='Reference input CSV file (CUDA results)'
    )
    parser.add_argument(
        '--f2',
        default='data/cases/details_stock_xpu.csv',
        help='Stock XPU input CSV file'
    )
    parser.add_argument(
        '--f3',
        default='data/cases/details_xpu-ops.csv',
        help='XPU-Ops input CSV file'
    )
    parser.add_argument(
        '--f4',
        default='data/cases/details_distributed.csv',
        help='Distributed XPU-Ops input CSV file'
    )

    # Filtering options
    parser.add_argument(
        '--include',
        dest='match',
        help='Path pattern to include (substring match)'
    )
    parser.add_argument(
        '--exclude',
        dest='notmatch',
        help='Path pattern to exclude (substring match)'
    )

    # Output options
    parser.add_argument(
        '-o', '--output',
        default='merged_details.csv',
        help='Output CSV file'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split output into multiple files for large datasets'
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

    try:
        merger = TestResultMerger()
        merger.merge(
            f1=args.f1,
            f2=args.f2,
            f3=args.f3,
            f4=args.f4,
            output=args.output,
            include_pattern=args.match,
            exclude_pattern=args.notmatch,
            split_output=args.split
        )
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise


if __name__ == "__main__":
    main()
