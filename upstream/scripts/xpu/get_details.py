#!/usr/bin/env python3
"""
Enhanced JUnit XML Test Details Extractor

Extracts test details from JUnit XML files and saves to Excel/CSV with improved
efficiency, error handling, and extensibility.

Usage:
    python get_details.py --input "results/*.xml" --output details.xlsx
    python get_details.py --input file1.xml file2.xml --output details.csv
    python get_details.py --input results_dir --output test_details.xlsx
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import logging
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar
from xml.etree import ElementTree as ET

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Type aliases
T = TypeVar("T")
Element = ET.Element
PathLike = str | Path


class TestStatus(Enum):
    """Standardized test status enumeration."""
    PASSED = "passed"
    XFAIL = "xfail"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    NOTRUN = ""
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, status_str: str) -> TestStatus:
        """Convert string to TestStatus enum."""
        if not status_str or pd.isna(status_str):
            return cls.NOTRUN

        status_str = str(status_str).lower().strip()

        status_mapping = {
            "pass": cls.PASSED,
            "success": cls.PASSED,
            "xfail": cls.XFAIL,
            "fail": cls.FAILED,
            "error": cls.ERROR,
            "skip": cls.SKIPPED,
            "": cls.NOTRUN,
        }

        for key, status in status_mapping.items():
            if key in status_str:
                return status

        return cls.UNKNOWN

    @property
    def priority(self) -> int:
        """Get priority for deduplication (higher = more important)."""
        priorities = {
            self.PASSED: 6,
            self.XFAIL: 5,
            self.FAILED: 4,
            self.ERROR: 3,
            self.SKIPPED: 2,
            self.UNKNOWN: 1,
            self.NOTRUN: 0,
        }
        return priorities[self]


class TestDevice(Enum):
    """Test device enumeration."""
    CUDA = "cuda"
    XPU = "xpu"
    UNKNOWN = "unknown"

    @classmethod
    def from_test_type(cls, test_type: str) -> TestDevice:
        """Extract device from test type string."""
        test_type_lower = test_type.lower()
        if "cuda" in test_type_lower:
            return cls.CUDA
        elif "xpu" in test_type_lower:
            return cls.XPU
        return cls.UNKNOWN


@dataclass(frozen=True, slots=True)
class TestCase:
    """Immutable data class representing a single test case."""

    # Core identifiers
    uniqname: str
    testfile: str
    classname: str
    name: str

    # Test properties
    device: TestDevice
    testtype: str
    status: TestStatus
    time: float

    # Metadata
    message: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestCase:
        """Create TestCase from dictionary with type conversion."""
        return cls(
            uniqname=data["uniqname"],
            testfile=data["testfile"],
            classname=data["classname"],
            name=data["name"],
            device=TestDevice(data["device"]) if isinstance(data["device"], str) else data["device"],
            testtype=data["testtype"],
            status=TestStatus(data["status"]) if isinstance(data["status"], str) else data["status"],
            time=float(data["time"]),
            message=data.get("message", ""),
        )

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        data = asdict(self)
        # Convert enums to strings
        data["device"] = data["device"].value
        data["status"] = data["status"].value
        return pd.Series(data)


class ProcessingStrategy(ABC):
    """Strategy pattern for different processing approaches."""

    @abstractmethod
    def process_files(self, extractor: TestDetailsExtractor, xml_files: List[Path]) -> None:
        """Process XML files using this strategy."""
        pass


class ParallelProcessingStrategy(ProcessingStrategy):
    """Parallel processing strategy using ThreadPoolExecutor."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def process_files(self, extractor: TestDetailsExtractor, xml_files: List[Path]) -> None:
        """Process files in parallel."""
        logger.info(f"Processing {len(xml_files)} files with {self.max_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(extractor._process_single_xml, xml_file): xml_file
                for xml_file in xml_files
            }

            with tqdm(total=len(xml_files), desc="Processing files", unit="file") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    xml_file = future_to_file[future]
                    try:
                        test_cases, metadata = future.result()
                        extractor._handle_processing_result(xml_file, test_cases, metadata)
                    except Exception as e:
                        logger.error(f"Error processing result for {xml_file}: {e}")
                        extractor.failed_files[str(xml_file)] = str(e)
                    finally:
                        extractor.stats["files_processed"] += 1
                        pbar.update(1)


class SequentialProcessingStrategy(ProcessingStrategy):
    """Sequential processing strategy for debugging or small datasets."""

    def process_files(self, extractor: TestDetailsExtractor, xml_files: List[Path]) -> None:
        """Process files sequentially."""
        logger.info(f"Processing {len(xml_files)} files sequentially")

        for xml_file in tqdm(xml_files, desc="Processing files", unit="file"):
            test_cases, metadata = extractor._process_single_xml(xml_file)
            extractor._handle_processing_result(xml_file, test_cases, metadata)
            extractor.stats["files_processed"] += 1


class FilePatternMatcher:
    """Handles file pattern matching and normalization."""

    # Compiled regex patterns for better performance
    _CLASSNAME_PATTERN = re.compile(r".*\.")
    _CASENAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")
    _TESTFILE_PATTERN = re.compile(r".*torch-xpu-ops\.test\.xpu\.")
    _TESTFILE_PATTERN_CPP = re.compile(r".*/test/xpu/")
    _NORMALIZE_PATTERN = re.compile(r".*\.\./test/")
    _GPU_PATTERN = re.compile(r"(?:xpu|cuda)", re.IGNORECASE)

    # Test type detection patterns
    TEST_TYPE_PATTERNS = {
        "xpu-default": [r"-test-default-.*-linux\.idc\.xpu_"],
        "xpu-unknown": [r"linux\.idc\.xpu_"],
        "cuda-default": [r"-test-default-.*(nvidia\.gpu|dgx\.)"],
        "cuda-inductor": [r"-test-inductor-.*(nvidia\.gpu|dgx\.)"],
        "cuda-distributed": [r"-test-distributed-.*(nvidia\.gpu|dgx\.)"],
        "cuda-unknown": [r"(nvidia\.gpu|dgx\.)"],
        "xpu-distributed": [r"xpu_distributed"],
        "xpu-ops": [r"op_ut_with_"],
    }

    # File replacement mappings
    FILE_REPLACEMENTS = [
        ("test/test/", "test/"),
        ("test_c10d_xccl.py", "test_c10d_nccl.py"),
        ("test_c10d_ops_xccl.py", "test_c10d_ops_nccl.py"),
    ]

    def __init__(self):
        self._compiled_test_type_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile all regex patterns for better performance."""
        return {
            test_type: [re.compile(pattern) for pattern in patterns]
            for test_type, patterns in self.TEST_TYPE_PATTERNS.items()
        }

    @lru_cache(maxsize=128)
    def determine_test_type(self, xml_file: Path) -> str:
        """Determine test type based on XML file path patterns."""
        xml_file_str = str(xml_file)

        for test_type, patterns in self._compiled_test_type_patterns.items():
            if any(pattern.search(xml_file_str) for pattern in patterns):
                return test_type

        return "xpu-undefied"

    def normalize_filepath(self, filepath: str, testtype: str) -> str:
        """Normalize test file path with replacements."""
        if not filepath:
            return "unknown_file.py"

        normalized = filepath

        # Apply regex normalization
        if self._NORMALIZE_PATTERN.search(normalized):
            normalized = self._NORMALIZE_PATTERN.sub("test/", normalized)

        # Apply string replacements
        for old, new in self.FILE_REPLACEMENTS:
            normalized = normalized.replace(old, new)

        if testtype in ['xpu-dist', 'xpu-ops']:
            normalized = normalized.replace("_xpu_xpu.py", ".py").replace("_xpu.py", ".py")

        return normalized

    def extract_testfile(self, classname: str, filename: str, xml_file: Path, testtype: str) -> str:
        """Extract and normalize test file path."""
        # Priority 1: Use filename from XML
        if filename:
            if filename.endswith(".cpp"):
                testfile = self._TESTFILE_PATTERN_CPP.sub("test/", filename)
            elif filename.endswith(".py"):
                testfile = f"test/{filename}"
            else:
                testfile = filename
        # Priority 2: Extract from classname
        elif classname:
            testfile = self._TESTFILE_PATTERN.sub("test/", classname).replace(".", "/")
            if "/" in testfile:
                testfile = f"{testfile.rsplit('/', 1)[0]}.py"
            else:
                testfile = f"{testfile}.py"
        # Priority 3: Extract from XML filename
        else:
            xml_file_str = str(xml_file)
            testfile = (
                re.sub(r".*op_ut_with_[a-zA-Z0-9]+\.", "test.", xml_file_str)
                .replace(".", "/")
                .replace("/py/xml", ".py")
                .replace("/xml", ".py")
            )

        return self.normalize_filepath(testfile, testtype)

    def extract_classname(self, full_classname: str) -> str:
        """Extract simplified classname from full classname."""
        if not full_classname:
            return "UnknownClass"
        return self._CLASSNAME_PATTERN.sub("", full_classname)

    def extract_casename(self, casename: str) -> str:
        """Extract normalized test case name."""
        if not casename:
            return "unknown_name"

        try:
            casename = self._CASENAME_PATTERN.sub("", casename)
            casename = self._TESTFILE_PATTERN.sub("", casename)
            return casename or "error_name"
        except Exception:
            return "error_name"

    def generate_uniqname(self, filename: str, classname: str, name: str) -> str:
        """Generate unique identifier for test case."""
        combined = f"{filename}{classname}{name}"
        return self._GPU_PATTERN.sub("cuda", combined)


def timer(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
        return result
    return wrapper

def load_last_details(input_file: str, sheet_name: list) -> pd.DataFrame:
    """
    Load test details from Excel or CSV file.
    """
    try:
        input_path = Path(input_file)
        print(f"Loading data from {input_file}")
        if input_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file, sheet_name=sheet_name, engine='openpyxl')
        else:
            df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} test cases from {input_file}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading details file {input_file}: {e}")
        raise

class TestDetailsExtractor:
    """
    Enhanced extractor for test details from JUnit XML files.

    Features:
    - Strategy pattern for processing
    - Comprehensive error handling
    - Memory-efficient streaming
    - Configurable via dependency injection
    """

    def __init__(
        self,
        processing_strategy: ProcessingStrategy | None = None,
        pattern_matcher: FilePatternMatcher | None = None,
        max_workers: int | None = None,
        use_cache: bool = True,
    ):
        """
        Initialize the extractor.

        Args:
            processing_strategy: Strategy for processing files
            pattern_matcher: File pattern matcher instance
            max_workers: Maximum number of parallel workers
            use_cache: Whether to use caching for expensive operations
        """
        self.pattern_matcher = pattern_matcher or FilePatternMatcher()
        self.processing_strategy = processing_strategy or ParallelProcessingStrategy(max_workers)
        self.use_cache = use_cache

        self.test_cases: List[TestCase] = []
        self.processed_files: List[str] = []
        self.empty_files: List[str] = []
        self.failed_files: Dict[str, str] = {}

        # Statistics
        self.stats = {
            "files_processed": 0,
            "test_cases_found": 0,
            "processing_time": 0.0,
        }

    @contextmanager
    def _measure_time(self, operation: str) -> Generator[None, None, None]:
        """Context manager for timing operations."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            logger.debug(f"{operation} completed in {elapsed:.3f}s")

    def _determine_test_status(self, testcase: Element) -> Tuple[TestStatus, str, str]:
        """Determine test status and extract message/type."""
        # Check for failure
        failure = testcase.find("failure")
        if failure is not None:
            message = failure.get("message", "")
            if "pytest.xfail" in message:
                return TestStatus.XFAIL, message, "xfail"
            return TestStatus.FAILED, message, "failure"

        # Check for skip
        skipped = testcase.find("skipped")
        if skipped is not None:
            message = skipped.get("message", "")
            skip_type = skipped.get("type", "")
            if "pytest.xfail" in skip_type or "pytest.xfail" in message:
                return TestStatus.XFAIL, message, "xfail"
            return TestStatus.SKIPPED, message, skip_type

        # Check for error
        error = testcase.find("error")
        if error is not None:
            message = error.get("message", "")
            return TestStatus.ERROR, message, "error"

        return TestStatus.PASSED, "", ""

    def _parse_testcase_element(self, testcase: Element, xml_file: Path) -> Optional[TestCase]:
        """Parse a single testcase element into a TestCase object."""
        try:
            classname = testcase.get("classname", "")
            filename = testcase.get("file", "")
            name = testcase.get("name", "")
            time_str = testcase.get("time", "0")

            # Determine test type
            test_type = self.pattern_matcher.determine_test_type(xml_file)

            # Extract and normalize
            simplified_classname = self.pattern_matcher.extract_classname(classname)
            simplified_casename = self.pattern_matcher.extract_casename(name)
            testfile = self.pattern_matcher.extract_testfile(classname, filename, xml_file, test_type)

            # Generate unique identifier
            uniqname = self.pattern_matcher.generate_uniqname(testfile, simplified_classname, simplified_casename)

            # Determine status
            status, message, result_type = self._determine_test_status(testcase)
            device = TestDevice.from_test_type(test_type)

            # Convert time to float safely
            try:
                time_val = float(time_str)
            except ValueError:
                time_val = 0.0

            return TestCase(
                uniqname=uniqname,
                testfile=testfile,
                classname=simplified_classname,
                name=simplified_casename,
                device=device,
                testtype=test_type,
                status=status,
                time=time_val,
                message=message,
            )

        except Exception as e:
            logger.error(f"Error parsing test case in {xml_file}: {e}")
            return None

    def find_xml_files(self, input_paths: List[PathLike]) -> List[Path]:
        """Find all XML files from various input specifications."""
        xml_files: Set[Path] = set()

        for input_path in input_paths:
            path = Path(input_path).expanduser().resolve()

            if path.is_file() and path.suffix.lower() == ".xml":
                xml_files.add(path)
            elif path.is_dir():
                # Use rglob for recursive search
                xml_files.update(path.rglob("*.xml"))
            else:
                # Handle glob patterns
                for file_path in glob.glob(str(path), recursive=True):
                    file_path = Path(file_path)
                    if file_path.is_file() and file_path.suffix.lower() == ".xml":
                        xml_files.add(file_path.resolve())

        return sorted(xml_files)

    def _process_single_xml(self, xml_file: Path) -> Tuple[List[TestCase], Dict[str, Any]]:
        """Process a single XML file and return test cases."""
        try:
            with self._measure_time(f"Parse {xml_file.name}"):
                tree = ET.parse(xml_file)
                root = tree.getroot()

            test_cases_elements = root.findall(".//testcase")

            if not test_cases_elements:
                return [], {"status": "empty", "file": str(xml_file)}

            test_cases = []
            for testcase in test_cases_elements:
                parsed_case = self._parse_testcase_element(testcase, xml_file)
                if parsed_case:
                    test_cases.append(parsed_case)

            return test_cases, {
                "status": "success",
                "file": str(xml_file),
                "count": len(test_cases),
            }

        except ET.ParseError as e:
            logger.error(f"XML parse error in {xml_file}: {e}")
            return [], {"status": "parse_error", "file": str(xml_file), "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error processing {xml_file}: {e}")
            return [], {"status": "error", "file": str(xml_file), "error": str(e)}

    def _handle_processing_result(self, xml_file: Path, test_cases: List[TestCase], metadata: Dict[str, Any]) -> None:
        """Handle the result of processing a single XML file."""
        if metadata["status"] == "empty":
            self.empty_files.append(str(xml_file))
        elif metadata["status"] == "success":
            self.test_cases.extend(test_cases)
            self.processed_files.append(str(xml_file))
            self.stats["test_cases_found"] += len(test_cases)
        else:
            self.failed_files[str(xml_file)] = metadata.get("error", "Unknown error")

    @timer
    def process(self, input_paths: List[PathLike]) -> bool:
        """Main processing method."""
        start_time = time.perf_counter()

        try:
            # Find XML files
            xml_files = self.find_xml_files(input_paths)

            if not xml_files:
                logger.error("No XML files found")
                return False

            logger.info(f"Found {len(xml_files)} XML files")

            # Process files using the selected strategy
            self.processing_strategy.process_files(self, xml_files)

            # Calculate statistics
            self.stats["processing_time"] = time.perf_counter() - start_time

            # Log summary
            self._log_summary()

            return len(self.test_cases) > 0

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False

    def _log_summary(self) -> None:
        """Log processing summary."""
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Test cases found: {self.stats['test_cases_found']}")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f}s")
        logger.info(f"Empty files: {len(self.empty_files)}")
        logger.info(f"Failed files: {len(self.failed_files)}")

        if self.empty_files and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Empty files:")
            for file in self.empty_files[:5]:
                logger.debug(f"  - {file}")
            if len(self.empty_files) > 5:
                logger.debug(f"  ... and {len(self.empty_files) - 5} more")

        if self.failed_files and logger.isEnabledFor(logging.WARNING):
            logger.warning("Failed files:")
            for file, error in list(self.failed_files.items())[:3]:
                logger.warning(f"  - {file}: {error}")
            if len(self.failed_files) > 3:
                logger.warning(f"  ... and {len(self.failed_files) - 3} more")


class TestResultAnalyzer:
    """Analyze and manipulate test results."""

    def __init__(self, test_cases: List[TestCase], last_df: pd.DataFrame, reson_df: pd.DataFrame):
        self.test_cases = test_cases
        self.dataframe = self._create_dataframe()
        self.dataframe = self.merge_last_results(last_df)
        self.dataframe = self.merge_last_resons(reson_df)

    def merge_last_results(self, last_df: pd.DataFrame) -> pd.DataFrame:
        """Merge CUDA and XPU test results for comparison."""
        if last_df.empty:
            return self.dataframe
        
        last_df_clean = last_df[["uniqname", "testfile", "classname", "name", "device", "testtype", "status", "time"]].copy()
        last_df_clean = last_df_clean.rename(columns={
            "testtype": "testtype_last",
            "status": "status_last",
            "time": "time_last"
        })
        
        # Merge with suffixes to distinguish current vs last results
        output = pd.merge(
            self.dataframe,
            last_df_clean,
            on=["uniqname", "testfile", "classname", "name", "device"],
            how="left",
            suffixes=("", "_last")  # Columns from last_df will get "_last" suffix
        )
        
        return output
    
    def merge_last_resons(self, reson_df: pd.DataFrame) -> pd.DataFrame:
        """Merge CUDA and XPU test results for comparison."""
        if reson_df.empty:
            return self.dataframe
        
        # Prepare reson_df
        reson_df_clean = reson_df[['Testfile', 'Class', 'Testcase', 'Reason', 'DetailReason']].copy()
        reson_df_clean = reson_df_clean.rename(columns={
            "Testfile": "testfile",
            "Class": "classname",
            "Testcase": "name",
        })
        reson_df_clean['device'] = 'cuda'
        # Merge
        output = pd.merge(
            self.dataframe,
            reson_df_clean,
            on=['device', 'testfile', 'classname', 'name'],
            how='left',
            suffixes=("", "_reason")  # Already added suffixes
        )
        
        return output

    def _create_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from test cases."""
        if not self.test_cases:
            return pd.DataFrame()

        # Use list comprehension for better performance
        data = [tc.to_series() for tc in self.test_cases]
        return pd.DataFrame(data)

    def get_unique_test_cases(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Get deduplicated test cases with status priority."""
        if df is None:
            df = self.dataframe.copy()
        if df.empty:
            return pd.DataFrame()

        # Add priority column using TestStatus enum
        df["_priority"] = df["status"].apply(
            lambda x: TestStatus.from_string(x).priority
        )

        # Group and select highest priority
        group_cols = ["device", "uniqname", "testfile", "classname", "name"]
        idx = df.groupby(group_cols)["_priority"].idxmax()

        result = df.loc[idx].drop("_priority", axis=1).reset_index(drop=True)

        return result

    def split_by_device(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame by device."""
        if df.empty or "device" not in df.columns:
            return pd.DataFrame(), pd.DataFrame()

        cuda_mask = df["device"] == "cuda"
        xpu_mask = df["device"] == "xpu"

        return df[cuda_mask].copy(), df[xpu_mask].copy()

    def merge_device_results(self, cuda_df: pd.DataFrame, xpu_df: pd.DataFrame) -> pd.DataFrame:
        """Merge CUDA and XPU test results for comparison."""
        if cuda_df.empty and xpu_df.empty:
            return pd.DataFrame()

        # Prepare dataframes with suffixes
        # cuda_prep = cuda_df.add_suffix("_cuda").rename(columns={"uniqname_cuda": "uniqname"})
        # xpu_prep = xpu_df.add_suffix("_xpu").rename(columns={"uniqname_xpu": "uniqname"})

        # Merge
        merged = pd.merge(
            cuda_df,
            xpu_df,
            on="uniqname",
            how="left",
            suffixes=("", "_xpu")  # Already added suffixes
        ).drop(columns=['Reason_xpu', 'DetailReason_xpu'])

        return merged

    def filter_by_pattern(self, df: pd.DataFrame, column: str, pattern: str, invert: bool = False) -> pd.DataFrame:
        """Filter DataFrame by pattern in a column."""
        if df.empty or column not in df.columns:
            return pd.DataFrame()

        mask = df[column].str.contains(pattern, case=False, na=False, regex=True)
        if invert:
            mask = ~mask

        return df[mask].copy()

    def get_xpu_only_skipped(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Get tests where CUDA passed but XPU failed/skipped."""
        if merged_df.empty:
            return pd.DataFrame()

        # Define conditions
        cuda_passed = merged_df["status"] == "passed"
        xpu_not_passed = (
            ~merged_df["status_xpu"].isin(["passed", "xfail"]) |
            merged_df["status_xpu"].isna()
        )

        return merged_df[cuda_passed & xpu_not_passed].copy()

    def generate_statistics(self, df: pd.DataFrame | None = None) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        df = df if df is not None else self.get_unique_test_cases()

        if df.empty:
            return {}

        stats = {
            "total_unique_cases": len(df),
            "by_device": {},
            "by_status": {},
            "by_test_type": {},
        }

        # Device statistics
        if "device" in df.columns:
            stats["by_device"] = df["device"].value_counts().to_dict()

        # Status statistics
        if "status" in df.columns:
            stats["by_status"] = df["status"].value_counts().to_dict()

        # Test type statistics
        if "testtype" in df.columns:
            stats["by_test_type"] = df["testtype"].value_counts().to_dict()

        # Time statistics
        if "time" in df.columns:
            stats["time_stats"] = {
                "total": df["time"].sum(),
                "mean": df["time"].mean(),
                "median": df["time"].median(),
                "max": df["time"].max(),
                "min": df["time"].min(),
                "std": df["time"].std(),
            }

        return stats


class ReportExporter:
    """Export test results to various formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_excel(self, analyzer: TestResultAnalyzer, output_path: Path) -> Dict[str, Path]:
        """Export results to Excel format."""
        output_files = {"main": output_path}

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Get unique test cases
            unique_df = analyzer.get_unique_test_cases()
            unique_df.to_excel(writer, sheet_name="All Test Cases", index=False)

            # Split by device
            cuda_df, xpu_df = analyzer.split_by_device(unique_df)

            if not cuda_df.empty and not xpu_df.empty:
                # Generate merged results
                merged_df = analyzer.merge_device_results(cuda_df, xpu_df)
                inductor_merged = analyzer.filter_by_pattern(merged_df, "testfile", "/inductor/")
                non_inductor_merged = analyzer.filter_by_pattern(merged_df, "testfile", "/inductor/", invert=True)

                inductor_merged.to_excel(writer, sheet_name="Merged Inductor", index=False)
                non_inductor_merged.to_excel(writer, sheet_name="Merged Non-Inductor", index=False)

                # Get XPU issues
                skipped_df = analyzer.get_xpu_only_skipped(merged_df)
                inductor_skipped = analyzer.filter_by_pattern(skipped_df, "testfile", "/inductor/")
                non_inductor_skipped = analyzer.filter_by_pattern(skipped_df, "testfile", "/inductor/", invert=True)

                inductor_skipped.to_excel(writer, sheet_name="XPU skipped only Inductor", index=False)
                non_inductor_skipped.to_excel(writer, sheet_name="XPU skipped only Non-Inductor", index=False)

            # Add statistics sheet
            stats = analyzer.generate_statistics(unique_df)
            if stats:
                stats_df = pd.DataFrame([stats])
                stats_df.to_excel(writer, sheet_name="Statistics", index=False)

        return output_files

    def export_csv(self, analyzer: TestResultAnalyzer, output_path: Path) -> Dict[str, Path]:
        """Export results to CSV format."""
        output_files = {"main": output_path}

        # Get unique test cases
        unique_df = analyzer.get_unique_test_cases()
        unique_df.to_csv(output_path, index=False)

        # Split by device
        cuda_df, xpu_df = analyzer.split_by_device(unique_df)

        if not cuda_df.empty and not xpu_df.empty:
            # Generate merged results
            merged_df = analyzer.merge_device_results(cuda_df, xpu_df)
            inductor_merged = analyzer.filter_by_pattern(merged_df, "testfile", "/inductor/")
            non_inductor_merged = analyzer.filter_by_pattern(merged_df, "testfile", "/inductor/", invert=True)

            # Get XPU issues
            skipped_df = analyzer.get_xpu_only_skipped(merged_df)
            inductor_skipped = analyzer.filter_by_pattern(skipped_df, "testfile", "/inductor/")
            non_inductor_skipped = analyzer.filter_by_pattern(skipped_df, "testfile", "/inductor/", invert=True)

            # Save additional files
            base_stem = output_path.stem
            suffixes = {
                "merged-inductor": inductor_merged,
                "merged-non-inductor": non_inductor_merged,
                "skipped-inductor": inductor_skipped,
                "skipped-non-inductor": non_inductor_skipped,
            }

            for suffix, df in suffixes.items():
                if not df.empty:
                    filename = f"{base_stem}_{suffix}{output_path.suffix}"
                    filepath = self.output_dir / filename
                    df.to_csv(filepath, index=False)
                    output_files[suffix] = filepath

            # Save statistics
            stats = analyzer.generate_statistics(unique_df)
            if stats:
                stats_file = self.output_dir / f"{base_stem}_stats.json"
                with open(stats_file, "w") as f:
                    json.dump(stats, f, indent=2, default=str)
                output_files["stats"] = stats_file

        return output_files

    def export(self, analyzer: TestResultAnalyzer, output_file: str) -> Dict[str, Path]:
        """Export all report types."""
        output_path = Path(output_file)

        if output_path.suffix.lower() in [".xlsx", ".xls"]:
            return self.export_excel(analyzer, output_path)
        else:
            return self.export_csv(analyzer, output_path)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced JUnit XML Test Details Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process multiple files
  python get_details.py --input results1.xml results2.xml --output details.xlsx

  # Process directory recursively
  python get_details.py --input results/ --output details.csv

  # Use glob pattern with parallel processing
  python get_details.py --input "results/**/*.xml" --output test_details.xlsx --parallel

  # Debug mode with sequential processing
  python get_details.py --input test.xml --output debug.csv --sequential --verbose
        """,
    )

    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="XML file paths, directories, or glob patterns",
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path (.xlsx, .xls, .csv)",
    )

    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory (default: current directory)",
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Use parallel processing (default: True)",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential processing (disables parallel)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with extra logging",
    )

    parser.add_argument(
        "--last",
        default=None,
        help="Last input file path (.xlsx, .xls, .csv)",
    )
    
    parser.add_argument(
        "--inductor",
        default=None,
        help="inductor input file path (.xlsx, .xls, .csv)",
    )
    
    parser.add_argument(
        "--non-inductor",
        default=None,
        help="non-inductor input file path (.xlsx, .xls, .csv)",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.getLogger().setLevel(log_level)

    # Determine processing strategy
    if args.sequential:
        processing_strategy = SequentialProcessingStrategy()
    else:
        processing_strategy = ParallelProcessingStrategy(max_workers=args.workers)

    try:
        # Load last details
        try:
            last_df = reson_df = pd.DataFrame()
            dataframes_to_concat = []
            
            if args.last is not None:
                last_df = load_last_details(args.last, ['All Test Cases'])
                last_df = last_df['All Test Cases']
            
            if args.inductor is not None:
                last_inductor_dfs = load_last_details(args.inductor, ['Cuda pass xpu skip', 'to_be_enabled'])
                dataframes_to_concat.extend(last_inductor_dfs.values())
            
            if args.non_inductor is not None:
                last_non_inductor_dfs = load_last_details(args.non_inductor, ['Non-Inductor XPU Skip'])
                dataframes_to_concat.append(last_non_inductor_dfs['Non-Inductor XPU Skip'])
            
            # Combine all DataFrames if we have any
            if dataframes_to_concat:
                reson_df = pd.concat(dataframes_to_concat, ignore_index=True)
                
        except Exception as e:
            logger.error(f"Failed to load details file: {e}")

        # Initialize extractor
        extractor = TestDetailsExtractor(
            processing_strategy=processing_strategy,
            max_workers=args.workers,
        )

        # Process files
        logger.info("Starting test details extraction...")
        success = extractor.process(args.input)

        if not success:
            logger.error("Extraction failed or no test cases found")
            return 1

        # Analyze results
        analyzer = TestResultAnalyzer(extractor.test_cases, last_df=last_df, reson_df=reson_df)

        # Export reports
        exporter = ReportExporter(Path(args.output_dir))
        output_files = exporter.export(analyzer, args.output)

        # Print summary
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"📊 Files processed: {extractor.stats['files_processed']}")
        print(f"🧪 Test cases found: {extractor.stats['test_cases_found']}")
        print(f"⏱️  Processing time: {extractor.stats['processing_time']:.2f}s")
        print(f"📈 Unique test cases: {len(analyzer.get_unique_test_cases())}")

        # Show device distribution
        unique_df = analyzer.get_unique_test_cases()
        if not unique_df.empty and "device" in unique_df.columns:
            device_counts = unique_df["device"].value_counts()
            print(f"📱 Device distribution:")
            for device, count in device_counts.items():
                print(f"   - {device}: {count}")

        print(f"📁 Output files:")
        for key, path in output_files.items():
            size_mb = path.stat().st_size / 1024 / 1024 if path.exists() else 0
            print(f"   - {key}: {path} ({size_mb:.2f} MB)")

        if extractor.empty_files:
            print(f"⚠️  Empty files: {len(extractor.empty_files)}")

        if extractor.failed_files:
            print(f"❌ Failed files: {len(extractor.failed_files)}")
            if args.verbose:
                for file, error in list(extractor.failed_files.items())[:3]:
                    print(f"   - {file}: {error}")

        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
