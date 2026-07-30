#!/usr/bin/env python3
"""
JUnit XML Test Details Extractor

Usage:
    python get_details.py --input "results/*.xml" --output details.xlsx
    python get_details.py --input file1.xml file2.xml --output details.csv
    python get_details.py --input results_dir --output test_details.xlsx --profile
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import glob
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypeVar, Union
from xml.etree import ElementTree as ET
from xml.parsers.expat import ExpatError

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
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
PathLike = Union[str, Path]


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class XMLProcessingError(Exception):
    """Base exception for XML processing errors."""
    pass

class TestCaseValidationError(XMLProcessingError):
    """Raised when test case validation fails."""
    pass

class InvalidXMLStructureError(XMLProcessingError):
    """Raised when XML structure is invalid."""
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

class ExtractionConfig(BaseModel):
    """Configuration for test extraction."""

    max_workers: int = Field(default=8, ge=1, le=64, description="Maximum parallel workers")
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Chunk size for processing")
    use_cache: bool = Field(default=True, description="Enable caching for expensive operations")
    cache_maxsize: int = Field(default=2048, ge=128, le=8192, description="Maximum cache size")
    default_output_format: str = Field(default="excel", pattern="^(excel|csv)$", description="Output format")
    logging_level: str = Field(default="INFO", description="Logging level")

    @field_validator('logging_level')
    @classmethod
    def validate_logging_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Logging level must be one of {valid_levels}")
        return v_upper

    model_config = ConfigDict(
        env_prefix="TEST_EXTRACTOR_",
        case_sensitive=False,
        extra="forbid",
        validate_assignment=True,
    )


# ============================================================================
# ENUMS
# ============================================================================

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
            "xpu-default": cls.PASSED,
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


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclasses.dataclass(frozen=True)
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

    def __post_init__(self) -> None:
        """Validate test case data after initialization."""
        if not self.uniqname or not isinstance(self.uniqname, str):
            raise TestCaseValidationError(f"Invalid uniqname: {self.uniqname}")

        if not self.testfile or not isinstance(self.testfile, str):
            raise TestCaseValidationError(f"Invalid testfile: {self.testfile}")

        if self.time < 0:
            raise TestCaseValidationError(f"Negative time value: {self.time}")

        if not isinstance(self.status, TestStatus):
            raise TestCaseValidationError(f"Invalid status type: {type(self.status)}")

        if not isinstance(self.device, TestDevice):
            raise TestCaseValidationError(f"Invalid device type: {type(self.device)}")


# ============================================================================
# UTILITIES
# ============================================================================

def timer_with_stats(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure execution time and track statistics."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            wrapper.total_time = getattr(wrapper, 'total_time', 0) + elapsed
            wrapper.call_count = getattr(wrapper, 'call_count', 0) + 1
            wrapper.avg_time = wrapper.total_time / wrapper.call_count
            logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper


def load_last_details(input_file: str, sheet_name: list) -> pd.DataFrame:
    """
    Load test details from Excel or CSV file.

    Args:
        input_file: Path to input file
        sheet_name: List of sheet names to load (for Excel)

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is not supported
    """
    try:
        input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logger.info(f"Loading data from {input_file}")

        if input_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file, sheet_name=sheet_name, engine='openpyxl')
        elif input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        logger.info(f"Loaded {len(df)} test cases from {input_file}")
        return df

    except Exception as e:
        logger.error(f"Error loading details file {input_file}: {e}")
        raise


# ============================================================================
# STRATEGY PATTERN
# ============================================================================

class ProcessingStrategy(ABC):
    """Strategy pattern for different processing approaches."""

    @abstractmethod
    def process_files(self, extractor: TestDetailsExtractor, xml_files: List[Path]) -> None:
        """Process XML files using this strategy."""
        pass


class ParallelProcessingStrategy(ProcessingStrategy):
    """Parallel processing strategy using ThreadPoolExecutor."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 4) + 4)

    @timer_with_stats
    def process_files(self, extractor: TestDetailsExtractor, xml_files: List[Path]) -> None:
        """Process files in parallel."""
        logger.info(f"Processing {len(xml_files)} files with {self.max_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="TestProcessor"
        ) as executor:
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

    @timer_with_stats
    def process_files(self, extractor: TestDetailsExtractor, xml_files: List[Path]) -> None:
        """Process files sequentially."""
        logger.info(f"Processing {len(xml_files)} files sequentially")

        for xml_file in tqdm(xml_files, desc="Processing files", unit="file"):
            test_cases, metadata = extractor._process_single_xml(xml_file)
            extractor._handle_processing_result(xml_file, test_cases, metadata)
            extractor.stats["files_processed"] += 1


# ============================================================================
# FILE PATTERN MATCHER
# ============================================================================

class FilePatternMatcher:
    """Handles file pattern matching and normalization."""

    # Compiled regex patterns for better performance
    _CLASSNAME_PATTERN = re.compile(r".*\.")
    _CASENAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")
    _TESTFILE_PATTERN = re.compile(r".*torch-xpu-ops\.test\.xpu\.")
    _TESTFILE_PATTERN_CPP = re.compile(r".*/test/xpu/")
    _NORMALIZE_PATTERN = re.compile(r".*\.\./test/")
    _GPU_PATTERN = re.compile(r"(?:xpu|cuda)", re.IGNORECASE)
    # Collapse the xpu/cuda markers and the literal "device"/"devices" token
    # (any spelling/position, optional adjacent underscore) when building the
    # device-agnostic merge key. "cpu" is intentionally NOT collapsed: a
    # cpu-parametrised test reports the same name under both the CUDA and XPU
    # harnesses, so it already matches by its real uniqname and must not be
    # merged into unrelated cases.
    _DEVICE_MARKER_PATTERN = re.compile(r"_?(?:cuda|xpu|devices?)_?", re.IGNORECASE)
    # Stage-2c marker: only the xpu/cuda device tags (not the "device" token),
    # so a stage-2c match is strictly more specific than the stage-2d strip.
    _GPU_MARKER_PATTERN = re.compile(r"_?(?:cuda|xpu)_?", re.IGNORECASE)

    # Test type detection patterns. Ordered list (not a dict) because several
    # test types map to more than one pattern; the first entry whose pattern
    # matches wins, and evaluation stops there.
    TEST_TYPE_PATTERNS = [
        ("xpu-cpp_wrapper", [r"-test-inductor_cpp_wrapper.*linux\.idc\.xpu"]),
        ("xpu-inductor", [r"-test-inductor.*linux\.idc\.xpu"]),
        ("xpu-default", [r"-test-default.*linux\.idc\.xpu"]),
        ("xpu-inductor", [r"/stock_xpu.*/inductor/"]),
        ("xpu-cpp_wrapper", [r"/stock_xpu.*cpp_wrapper"]),
        ("xpu-distributed", [r"test/distributed/"]),
        ("xpu-distributed", [r"xpu_distributed"]),
        ("xpu-ops", [r"xpu-ops"]),
        ("xpu-bmg", [r"linux\.client\.xpu"]),
        ("xpu-unknown", [r"linux\.idc\.xpu"]),
        # Default (non-inductor) stock XPU suite, e.g. .../stock_xpu/xmls/*.xml;
        # kept after the inductor/cpp_wrapper stock_xpu rules so those win first.
        ("xpu-default", [r"/stock_xpu/"]),
        ("cuda-cpp_wrapper", [r"-test-inductor_cpp_wrapper.*(nvidia|linux.dgx)"]),
        ("cuda-cpp_wrapper", [r"/cuda.*/cpp_wrapper/"]),
        ("cuda-inductor", [r"-test-inductor.*(nvidia|linux.dgx)"]),
        ("cuda-inductor", [r"/cuda.*/inductor/"]),
        ("cuda-distributed", [r"-test-distributed.*(nvidia|linux.dgx)"]),
        ("cuda-distributed", [r"/cuda.*/distributed/"]),
        ("cuda-default", [r"-test-default.*(nvidia|linux.dgx)"]),
        ("cuda-distributed", [r"/cuda.*/default/"]),
        ("cuda-unknown", [r"(nvidia|linux.dgx)"]),
    ]

    # File replacement mappings
    FILE_REPLACEMENTS = [
        ("test/test/", "test/"),
        ("test_c10d_xccl.py", "test_c10d_nccl.py"),
        ("test_c10d_ops_xccl.py", "test_c10d_ops_nccl.py"),
    ]

    def __init__(self):
        self._compiled_test_type_patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[Tuple[str, List[re.Pattern]]]:
        """Compile all regex patterns for better performance."""
        return [
            (test_type, [re.compile(pattern) for pattern in patterns])
            for test_type, patterns in self.TEST_TYPE_PATTERNS
        ]

    @lru_cache(maxsize=2048)
    def determine_test_type(self, xml_file: Path) -> str:
        """Determine test type based on XML file path patterns."""
        xml_file_str = str(xml_file)

        for test_type, patterns in self._compiled_test_type_patterns:
            if any(pattern.search(xml_file_str) for pattern in patterns):
                return test_type

        return "others-undefined"

    @lru_cache(maxsize=2048)
    def normalize_filepath(self, filepath: str, testtype: str) -> str:
        """Normalize test file path with replacements (cached)."""
        if not filepath:
            return "unknown_file.py"

        normalized = filepath

        # Apply regex normalization
        if self._NORMALIZE_PATTERN.search(normalized):
            normalized = self._NORMALIZE_PATTERN.sub("test/", normalized)

        # Apply string replacements efficiently
        for old, new in self.FILE_REPLACEMENTS:
            if old in normalized:
                normalized = normalized.replace(old, new)

        if testtype in ['xpu-distributed', 'xpu-ops']:
            normalized = normalized.replace("_xpu_xpu.py", ".py").replace("_xpu.py", ".py")
        normalized = re.sub(r'.*/jenkins/workspace/', '', normalized, flags=re.IGNORECASE)

        return normalized or "unknown_file.py"

    @lru_cache(maxsize=1024)
    def extract_testfile(self, classname: str, filename: str, xml_file: Path, testtype: str) -> str:
        """Extract and normalize test file path (cached)."""
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

    @lru_cache(maxsize=1024)
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

    @lru_cache(maxsize=2048)
    def generate_uniqname(self, filename: str, classname: str, name: str) -> str:
        """Generate the *real-name* unique identifier for a test case.

        Per the merge design the uniqname preserves the real names: it is simply
        the concatenation of the test file basename, test class and test name.
        Device-marker normalisation is intentionally deferred
        to merge time (see :meth:`normalize_uniqname`) so the stored identifier
        stays faithful to the source while xpu/cuda/cpu variants can still be
        collapsed when comparing devices.
        """
        # Use the file basename so flattened XPU paths (e.g. test/test_distributions.py)
        # match nested CUDA paths (e.g. test/distributions/test_distributions.py).
        file_key = filename.rsplit("/", 1)[-1] if filename else filename
        return f"{file_key}{classname}{name}"

    @lru_cache(maxsize=2048)
    def normalize_uniqname(self, uniqname: str) -> str:
        """Stage-2d device-agnostic key used to merge xpu/cuda cases.

        Collapses the xpu/cuda markers and the literal "device"/"devices" token
        (optional adjacent underscore, case-insensitive). "cpu" is deliberately
        left intact: a cpu-parametrised test reports the same name under both
        harnesses, so it already matches by its real uniqname and must not be
        folded into unrelated cases. This is the loosest stage (2d); stage 2c
        (:meth:`normalize_uniqname_gpu`) is tried first.
        """
        return self._DEVICE_MARKER_PATTERN.sub("", uniqname)

    @lru_cache(maxsize=2048)
    def normalize_uniqname_gpu(self, uniqname: str) -> str:
        """Stage-2c key: strip only the xpu/cuda markers (keep the rest).

        Tried before :meth:`normalize_uniqname` (2d) so a case that already
        pairs after removing just the xpu/cuda tags is matched at the more
        specific stage and never reconsidered at the looser "device" strip.
        """
        return self._GPU_MARKER_PATTERN.sub("", uniqname)


# ============================================================================
# TEST DETAILS EXTRACTOR
# ============================================================================

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
        processing_strategy: Optional[ProcessingStrategy] = None,
        pattern_matcher: Optional[FilePatternMatcher] = None,
        config: Optional[ExtractionConfig] = None,
    ):
        """
        Initialize the extractor.

        Args:
            processing_strategy: Strategy for processing files
            pattern_matcher: File pattern matcher instance
            config: Extraction configuration
        """
        self.config = config or ExtractionConfig()
        self.pattern_matcher = pattern_matcher or FilePatternMatcher()
        self.processing_strategy = processing_strategy or ParallelProcessingStrategy(
            max_workers=self.config.max_workers
        )

        # Apply cache configuration
        if self.config.use_cache:
            for method_name in ['normalize_filepath', 'extract_testfile',
                              'determine_test_type', 'generate_uniqname']:
                method = getattr(self.pattern_matcher, method_name)
                method.__dict__['maxsize'] = self.config.cache_maxsize

        self.test_cases: List[TestCase] = []
        self.processed_files: List[str] = []
        self.empty_files: List[str] = []
        self.failed_files: Dict[str, str] = {}

        # Statistics
        self.stats = {
            "files_processed": 0,
            "test_cases_found": 0,
            "processing_time": 0.0,
            "memory_used_mb": 0.0,
        }

    def _determine_test_status(self, testcase: Element) -> Tuple[TestStatus, str, str]:
        """Determine test status and extract message/type."""
        def extract_result_details(result: Element) -> Tuple[str, str, str]:
            message = (result.get("message") or "").strip()
            result_type = (result.get("type") or "").strip()
            text = "\n".join(
                part.strip() for part in result.itertext() if part and part.strip()
            )
            return message or text, result_type, text

        # Check for failure
        failure = testcase.find("failure")
        if failure is not None:
            message, failure_type, failure_text = extract_result_details(failure)
            failure_details = " ".join(
                part for part in [message, failure_type, failure_text] if part
            ).lower()
            if "pytest.xfail" in failure_details or "xfail" in failure_details:
                return TestStatus.XFAIL, message, "xfail"
            if "xpass" in failure_details:
                return TestStatus.FAILED, message, "xpass"
            return TestStatus.FAILED, message, failure_type or "failure"

        # Check for skip
        skipped = testcase.find("skipped")
        if skipped is not None:
            message, skip_type, skip_text = extract_result_details(skipped)
            skip_details = " ".join(
                part for part in [message, skip_type, skip_text] if part
            ).lower()
            if "pytest.xfail" in skip_details or "xfail" in skip_details:
                return TestStatus.XFAIL, message, "xfail"
            return TestStatus.SKIPPED, message, skip_type or "skipped"

        # Check for error
        error = testcase.find("error")
        if error is not None:
            message, error_type, _ = extract_result_details(error)
            return TestStatus.ERROR, message, error_type or "error"

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
            status, message, _ = self._determine_test_status(testcase)
            device = TestDevice.from_test_type(test_type)

            # Convert time to float safely
            try:
                time_val = float(time_str)
            except (ValueError, TypeError):
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
                message=message.replace('\n', '<tr>')[:200],
            )

        except TestCaseValidationError as e:
            logger.warning(f"Test case validation failed in {xml_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing test case in {xml_file}: {e}")
            return None

    def _process_xml_streaming(self, xml_file: Path) -> Generator[List[TestCase], None, None]:
        """Process XML file in chunks to reduce memory usage."""
        try:
            # Use iterparse for memory-efficient parsing
            context = ET.iterparse(xml_file, events=('start', 'end'))

            chunk: List[TestCase] = []
            for event, elem in context:
                if event == 'end' and elem.tag == 'testcase':
                    test_case = self._parse_testcase_element(elem, xml_file)
                    if test_case:
                        chunk.append(test_case)

                        if len(chunk) >= self.config.chunk_size:
                            yield chunk
                            chunk = []

                    # Clear element to free memory
                    elem.clear()

            # Yield remaining test cases
            if chunk:
                yield chunk

        except (ET.ParseError, ExpatError) as e:
            logger.error(f"XML parse error in {xml_file}: {e}")
            raise InvalidXMLStructureError(f"Failed to parse {xml_file}: {e}")

    @timer_with_stats
    def _process_single_xml(self, xml_file: Path) -> Tuple[List[TestCase], Dict[str, Any]]:
        """Process a single XML file and return test cases."""
        try:
            all_test_cases: List[TestCase] = []

            # Process in chunks for memory efficiency
            for chunk in self._process_xml_streaming(xml_file):
                all_test_cases.extend(chunk)

            if not all_test_cases:
                return [], {"status": "empty", "file": str(xml_file)}

            return all_test_cases, {
                "status": "success",
                "file": str(xml_file),
                "count": len(all_test_cases),
            }

        except InvalidXMLStructureError as e:
            logger.error(f"Invalid XML structure in {xml_file}: {e}")
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

    @timer_with_stats
    def process(self, input_paths: List[PathLike]) -> bool:
        """Main processing method."""
        import psutil

        start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

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
            end_time = time.perf_counter()
            final_memory = process.memory_info().rss / 1024 / 1024

            self.stats["processing_time"] = end_time - start_time
            self.stats["memory_used_mb"] = final_memory - initial_memory

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
        logger.info(f"Memory used: {self.stats['memory_used_mb']:.2f} MB")
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


# ============================================================================
# PYTORCH SOURCE RESOLVER
# ============================================================================

class PyTorchSourceResolver:
    """Best-effort resolver mapping a test case to its PyTorch source file.

    Used to reconcile CUDA/XPU results that fail to match by uniqname: the same
    test can live under a differently named file or class across device runs
    (so their generated uniqnames diverge). By locating the file that defines a
    given class/test in the PyTorch ``test`` tree (e.g. the ``release/2.13``
    branch), a canonical, device-independent uniqname can be recomputed so the
    CUDA and XPU rows collapse onto one identifier.

    All network access is best-effort and cached on disk; any failure (offline,
    rate limit, parse error) degrades gracefully to a no-op so the rest of the
    pipeline is unaffected. Set ``GITHUB_TOKEN``/``GH_TOKEN`` to raise the
    GitHub rate limit for the initial source download.
    """

    _TREE_URL = "https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
    _RAW_URL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    _CLASS_RE = re.compile(r"^[ \t]*class[ \t]+(\w+)", re.MULTILINE)
    _DEF_RE = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+(test\w*)", re.MULTILINE)
    # Collapse device markers so class/test names compare device-agnostically.
    _DEVICE_RE = re.compile(r"_?(?:cuda|xpu|cpu)_?", re.IGNORECASE)

    def __init__(
        self,
        ref: str,
        repo: str = "pytorch/pytorch",
        subtree: str = "test",
        enabled: bool = True,
        cache_dir: Optional[PathLike] = None,
        timeout: int = 30,
        max_workers: int = 8,
        token: Optional[str] = None,
    ):
        self.ref = ref
        self.repo = repo
        self.subtree = subtree.strip("/") + "/"
        self.enabled = bool(enabled and ref)
        self.timeout = timeout
        self.max_workers = max_workers
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

        base = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "get_details")
        self.cache_dir = base / "pytorch_src" / repo.replace("/", "_") / ref.replace("/", "_")

        self._indexed = False
        self._unavailable = False
        self._class_index: Dict[str, Set[str]] = defaultdict(set)
        self._def_index: Dict[str, Set[str]] = defaultdict(set)

    # -- normalization -----------------------------------------------------
    def _norm(self, text: str) -> str:
        """Lower-case and strip device markers for device-agnostic comparison."""
        return self._DEVICE_RE.sub("", text or "").lower()

    # -- HTTP --------------------------------------------------------------
    def _http_get(self, url: str) -> Optional[bytes]:
        """Fetch a URL, disabling the resolver on unrecoverable failures."""
        if self._unavailable:
            return None
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", "get_details-source-resolver")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                logger.warning(
                    "PyTorch source lookup rate-limited/forbidden (HTTP %s); "
                    "set GITHUB_TOKEN to raise the limit. Disabling resolver.",
                    e.code,
                )
                self._unavailable = True
            else:
                logger.debug("HTTP %s for %s", e.code, url)
            return None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            logger.warning("PyTorch source lookup failed (%s); disabling resolver.", e)
            self._unavailable = True
            return None

    # -- tree / content ----------------------------------------------------
    def _load_tree(self) -> List[str]:
        """Return the list of ``test/**/*.py`` paths, cached on disk."""
        cache = self.cache_dir / "tree.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                return [p for p in data if p.startswith(self.subtree) and p.endswith(".py")]
            except (OSError, json.JSONDecodeError):
                pass

        raw = self._http_get(self._TREE_URL.format(repo=self.repo, ref=self.ref))
        if not raw:
            return []
        try:
            tree = json.loads(raw).get("tree", [])
        except json.JSONDecodeError:
            return []

        paths = [
            item["path"]
            for item in tree
            if item.get("type") == "blob"
            and item.get("path", "").startswith(self.subtree)
            and item["path"].endswith(".py")
        ]
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(paths))
        except OSError:
            pass
        return paths

    def _content_cache_path(self, path: str) -> Path:
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()
        return self.cache_dir / "files" / f"{digest}.py"

    def _get_content(self, path: str) -> Optional[str]:
        """Return the raw source of ``path``, cached on disk."""
        cached = self._content_cache_path(path)
        if cached.exists():
            try:
                return cached.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return None

        raw = self._http_get(self._RAW_URL.format(repo=self.repo, ref=self.ref, path=path))
        if raw is None:
            return None
        text = raw.decode("utf-8", errors="ignore")
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(text, encoding="utf-8")
        except OSError:
            pass
        return text

    def _index_content(self, path: str, content: str) -> None:
        for cls in self._CLASS_RE.findall(content):
            self._class_index[self._norm(cls)].add(path)
        for fn in self._DEF_RE.findall(content):
            self._def_index[self._norm(fn)].add(path)

    def _build_index(self) -> None:
        """Download and index the PyTorch test tree once (lazily)."""
        if self._indexed or self._unavailable:
            return
        paths = self._load_tree()
        if not paths:
            self._unavailable = True
            self._indexed = True
            return

        logger.info("Indexing %d PyTorch test files for source reconciliation...", len(paths))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._get_content, p): p for p in paths}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Indexing sources",
                unit="file",
            ):
                path = futures[future]
                try:
                    content = future.result()
                except Exception:
                    content = None
                if content:
                    self._index_content(path, content)
        self._indexed = True

    # -- resolution --------------------------------------------------------
    def resolve_file(self, classname: str, name: str, testfile_hint: str = "") -> Optional[str]:
        """Return the canonical source file that defines ``classname``/``name``.

        Returns ``None`` when the resolver is disabled/unavailable or no
        matching definition can be found.
        """
        if not self.enabled or self._unavailable:
            return None
        self._build_index()
        if not self._class_index and not self._def_index:
            return None

        class_paths = self._class_index.get(self._norm(classname), set())

        # Progressive suffix trimming strips dtype/param tails so a reported
        # name like ``test_add_float32`` still matches source ``def test_add``.
        name_key = self._norm(name)
        def_paths: Set[str] = set()
        tokens = name_key.split("_")
        for i in range(len(tokens), 0, -1):
            candidate = "_".join(tokens[:i])
            if candidate in self._def_index:
                def_paths = self._def_index[candidate]
                break

        if class_paths and def_paths:
            candidates = (class_paths & def_paths) or class_paths
        elif class_paths:
            candidates = class_paths
        else:
            candidates = def_paths

        if not candidates:
            return None
        # Deterministic choice so CUDA and XPU rows resolve to the same file.
        return min(candidates)

    def canonical_uniqname(
        self,
        pattern_matcher: "FilePatternMatcher",
        classname: str,
        name: str,
        testfile_hint: str = "",
    ) -> Optional[str]:
        """Recompute a device-independent uniqname from the resolved source file."""
        path = self.resolve_file(classname, name, testfile_hint)
        if not path:
            return None
        return pattern_matcher.generate_uniqname(path, classname, name)


# ============================================================================
# TEST RESULT ANALYZER
# ============================================================================

class TestResultAnalyzer:
    """Analyze and manipulate test results."""

    def __init__(self, test_cases: List[TestCase], last_df: pd.DataFrame, reson_df: pd.DataFrame, pattern_matcher: Optional[FilePatternMatcher] = None, source_resolver: Optional[PyTorchSourceResolver] = None,):
        self.test_cases = test_cases
        self.pattern_matcher = pattern_matcher or FilePatternMatcher()
        self.source_resolver = source_resolver
        self.dataframe = self._create_dataframe_optimized()
        # Reconcile CUDA/XPU cases that did not match by uniqname against the
        # PyTorch test sources before merging historical/reason data.
        if self.source_resolver is not None and self.source_resolver.enabled:
            self.dataframe = self._reconcile_with_source()
        self.dataframe = self.merge_last_results_optimized(last_df)
        self.dataframe = self.merge_last_reasons_optimized(reson_df)

    def _reconcile_with_source(self) -> pd.DataFrame:
        """Merge xpu/cuda cases with different realnames via PyTorch source.

        After the exact-uniqname match (2a) fails, cases whose device-normalised
        keys still differ across devices are looked up in the PyTorch test tree;
        when the same defining source file is found for a CUDA and an XPU case,
        their uniqname is rewritten onto that canonical file so they collapse
        onto one identifier. This runs before the device-marker stages (2c/2d)
        and only ever adds merges -- already-matched pairs are left untouched.
        """
        df = self.dataframe
        if df.empty or "device" not in df.columns or "uniqname" not in df.columns:
            return df

        # Cases are considered matched across devices when their *device-
        # normalised* keys coincide, so ordinary xpu/cuda/cpu
        # variants never trigger a (costly) source lookup -- only genuinely
        # differently named cases do.
        norm = self.pattern_matcher.normalize_uniqname
        nk = df["uniqname"].map(norm)
        gpu_mask = df["device"].isin(["cuda", "xpu"])
        cuda_keys = set(nk[df["device"] == "cuda"])
        xpu_keys = set(nk[df["device"] == "xpu"])
        matched = cuda_keys & xpu_keys

        # Only rows that failed to match across devices are worth resolving.
        unmatched_mask = gpu_mask & ~nk.isin(matched)
        unmatched = df.loc[unmatched_mask, ["uniqname", "testfile", "classname", "name"]]
        if unmatched.empty:
            return df

        # Resolve each distinct case once; map old uniqname -> canonical uniqname.
        remap: Dict[str, str] = {}
        seen: Set[Tuple[str, str, str]] = set()
        for row in unmatched.itertuples(index=False):
            key = (row.classname, row.name, row.testfile)
            if key in seen:
                continue
            seen.add(key)
            canonical = self.source_resolver.canonical_uniqname(
                self.pattern_matcher, row.classname, row.name, row.testfile
            )
            if canonical and canonical != row.uniqname:
                remap[row.uniqname] = canonical

        if not remap:
            return df

        df.loc[unmatched_mask, "uniqname"] = (
            df.loc[unmatched_mask, "uniqname"]
            .map(remap)
            .fillna(df.loc[unmatched_mask, "uniqname"])
        )
        logger.info(
            "Reconciled %d uniqname(s) via PyTorch source (%s)",
            len(remap),
            self.source_resolver.ref,
        )
        return df


    def _create_dataframe_optimized(self) -> pd.DataFrame:
        """Create DataFrame from test cases with optimization."""
        if not self.test_cases:
            return pd.DataFrame()

        # Pre-allocate columns for better performance
        columns = ["uniqname", "testfile", "classname", "name", "device",
                   "testtype", "status", "time", "message"]

        # Use dictionary of lists for faster DataFrame creation
        data = {col: [] for col in columns}

        for tc in self.test_cases:
            data["uniqname"].append(tc.uniqname)
            data["testfile"].append(tc.testfile)
            data["classname"].append(tc.classname)
            data["name"].append(tc.name)
            data["device"].append(tc.device.value)
            data["testtype"].append(tc.testtype)
            data["status"].append(tc.status.value)
            data["time"].append(tc.time)
            data["message"].append(tc.message)

        return pd.DataFrame(data)

    def merge_last_results_optimized(self, last_df: pd.DataFrame) -> pd.DataFrame:
        """Merge the previous run's status/time onto the current cases.

        The join keys are the real identity fields
        ``testfile, classname, name`` plus ``device`` (device is required so a
        CPU case, which reports the same name under both the CUDA and XPU runs,
        does not fan out into a cross product).
        """
        keys = ["testfile", "classname", "name", "device"]
        if last_df.empty:
            return pd.concat([self.dataframe, pd.DataFrame(columns=['last_status', 'last_time'])], axis=1)

        # Keep the identity keys plus the fields carried over from the previous run.
        last_df_clean = last_df[keys + ["status", "time"]].copy()

        # Convert status to strings if they're enums
        if not last_df_clean['status'].empty and hasattr(last_df_clean['status'].iloc[0], 'value'):
            last_df_clean['status'] = last_df_clean['status'].apply(lambda x: x.value)

        last_df_clean = last_df_clean.rename(columns={
            "status": "last_status",
            "time": "last_time",
        })

        # Deduplicate previous results on the join keys to avoid row explosion.
        last_df_clean = last_df_clean.drop_duplicates(subset=keys, keep="first")

        merged = pd.merge(
            self.dataframe,
            last_df_clean,
            on=keys,
            how="outer",
            sort=False,
        )

        # Rows present only in the previous run have no uniqname yet; recompute a
        # consistent real-name uniqname for every row from its identity fields so
        # the "All Test Cases" dedup (on device + uniqname) stays well defined.
        gen = self.pattern_matcher.generate_uniqname
        triples = merged[["testfile", "classname", "name"]].fillna("").drop_duplicates()
        triples["uniqname"] = [
            gen(str(t), str(c), str(n))
            for t, c, n in zip(triples["testfile"], triples["classname"], triples["name"])
        ]
        if "uniqname" in merged.columns:
            merged = merged.drop(columns=["uniqname"])
        merged = merged.merge(triples, on=["testfile", "classname", "name"], how="left")

        return merged.fillna('')

    def merge_last_reasons_optimized(self, reson_df: pd.DataFrame) -> pd.DataFrame:
        """Attach the previous run's reasons, matched by uniqname.

        The reason's uniqname is built from cuda's realnames
        (``testfile_cuda``/``classname_cuda``/``name_cuda``) and reduced to the
        device-agnostic key, so it matches through the same xpu/cuda uniqname
        merge used elsewhere -- i.e. a cuda-derived reason lands on both the
        cuda and the xpu rows of the same case.
        """
        if reson_df.empty:
            return pd.concat([self.dataframe, pd.DataFrame(columns=['Reason', 'DetailReason'])], axis=1)

        gen = self.pattern_matcher.generate_uniqname
        norm = self.pattern_matcher.normalize_uniqname

        reson_df_clean = reson_df[['testfile_cuda', 'classname_cuda', 'name_cuda', 'Reason', 'DetailReason']].copy()
        # Reason uniqname comes from cuda's names, then normalised to the
        # device-agnostic key so it merges like the xpu/cuda case merge.
        reson_df_clean['_rkey'] = [
            norm(gen(str(t), str(c), str(n)))
            for t, c, n in zip(
                reson_df_clean['testfile_cuda'],
                reson_df_clean['classname_cuda'],
                reson_df_clean['name_cuda'],
            )
        ]
        reson_df_clean = (
            reson_df_clean[['_rkey', 'Reason', 'DetailReason']]
            .drop_duplicates(subset=['_rkey'], keep='first')
        )

        merged = self.dataframe.copy()
        merged['_rkey'] = merged['uniqname'].map(norm)
        merged = pd.merge(
            merged,
            reson_df_clean,
            on='_rkey',
            how='left',
            sort=False,
        ).drop(columns=['_rkey'])
        return merged.fillna('')

    def get_unique_test_cases(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Deduplicate to a single row per ``(device, uniqname)``.

        Among the rows sharing a ``(device, uniqname)`` the most representative
        one is kept: rows whose ``name``/``classname``/``testfile`` carry the
        device marker (i.e. the device-specific variant) win, and the
        highest-priority ``status`` breaks the final tie.
        """
        if df is None:
            df = self.dataframe.copy()
        if df.empty:
            return pd.DataFrame()

        gpu = self.pattern_matcher._GPU_PATTERN
        # Normalise the device marker (xpu/cuda -> cuda) once per column, then
        # flag rows whose field carries that marker with a vectorised substring
        # test (far cheaper than a row-wise ``apply``).
        device_norm = df["device"].astype(str).str.replace(gpu, "cuda", regex=True)
        for col in ("name", "classname", "testfile"):
            field_norm = df[col].astype(str).str.replace(gpu, "cuda", regex=True)
            df[f"_{col}"] = [needle in hay for needle, hay in zip(device_norm, field_norm)]
        df["_status"] = df["status"].map(lambda s: TestStatus.from_string(s).priority)

        # Highest-priority row first so ``keep="first"`` retains it; a stable
        # sort keeps the ordering deterministic for otherwise-equal rows.
        sort_cols = ["_name", "_classname", "_testfile", "_status"]
        df_sorted = df.sort_values(by=sort_cols, ascending=False, kind="stable")

        return (
            df_sorted
            .drop_duplicates(subset=["device", "uniqname"], keep="first")
            .drop(columns=["_name", "_classname", "_testfile", "_status"])
            .reset_index(drop=True)
        )

    def split_by_device(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame by device."""
        if df.empty or "device" not in df.columns:
            return pd.DataFrame(), pd.DataFrame()

        cuda_mask = df["device"] == "cuda"
        xpu_mask = df["device"] == "xpu"

        return df[cuda_mask].copy(), df[xpu_mask].copy()

    def _collapse_device_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse rows sharing the same ``_mergekey``, keeping the best status.

        ``df`` must already carry a ``_mergekey`` column assigned by
        :meth:`_staged_device_mergekeys`. Rows that a stage paired onto the same
        key are treated as one so the cross-device merge cannot fan out into
        duplicate rows.
        """
        if df.empty:
            return df if "_mergekey" in df.columns else df.assign(_mergekey=pd.Series(dtype=str))
        df = df.copy()
        df["_priority"] = df["status"].map(lambda s: TestStatus.from_string(s).priority)
        df = (
            df.sort_values("_priority", ascending=False, kind="stable")
            .drop_duplicates(subset=["_mergekey"], keep="first")
            .drop(columns=["_priority"])
            .reset_index(drop=True)
        )
        return df

    def _staged_device_mergekeys(
        self, cuda_df: pd.DataFrame, xpu_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Assign a shared ``_mergekey`` to matching cuda/xpu rows in stage order.

        Implements the a.txt staged merge with return-on-match: 2a (exact real
        uniqname) -> 2c (strip xpu/cuda) -> 2d (strip xpu/cuda + "device"). The
        first stage that pairs a cuda row with an xpu row wins and neither row is
        reconsidered at a looser stage. (2b source reconciliation runs earlier as
        a uniqname remap, so it manifests here through the remapped uniqname.)
        Rows that never pair keep a unique key and stay device-only (2e).
        """
        pm = self.pattern_matcher
        stages = [
            ("a", lambda u: u),
            ("c", pm.normalize_uniqname_gpu),
            ("d", pm.normalize_uniqname),
        ]
        cuda_df = cuda_df.reset_index(drop=True).copy()
        xpu_df = xpu_df.reset_index(drop=True).copy()
        cuda_df["_mergekey"] = pd.NA
        xpu_df["_mergekey"] = pd.NA

        for tag, fn in stages:
            c_res = cuda_df["_mergekey"].isna()
            x_res = xpu_df["_mergekey"].isna()
            # Matching needs a residual row on both sides; stop once either is done.
            if not c_res.any() or not x_res.any():
                break
            ckey = cuda_df.loc[c_res, "uniqname"].map(fn)
            xkey = xpu_df.loc[x_res, "uniqname"].map(fn)
            common = set(ckey) & set(xkey)
            if not common:
                continue
            c_hit = ckey[ckey.isin(common)]
            x_hit = xkey[xkey.isin(common)]
            # Encode the stage tag so a match at one stage can never share a key
            # with a match at another (looser) stage.
            cuda_df.loc[c_hit.index, "_mergekey"] = tag + "\x1f" + c_hit.astype(str)
            xpu_df.loc[x_hit.index, "_mergekey"] = tag + "\x1f" + x_hit.astype(str)

        # 2e: unmatched rows get a unique key so they never merge with anything.
        c_un = cuda_df["_mergekey"].isna()
        cuda_df.loc[c_un, "_mergekey"] = [f"uc\x1f{n}" for n in range(int(c_un.sum()))]
        x_un = xpu_df["_mergekey"].isna()
        xpu_df.loc[x_un, "_mergekey"] = [f"ux\x1f{n}" for n in range(int(x_un.sum()))]
        return cuda_df, xpu_df

    def merge_device_results(self, cuda_df: pd.DataFrame, xpu_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge CUDA and XPU test results for comparison."""
        if cuda_df.empty and xpu_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Staged merge (a.txt 2a>2b>2c>2d>2e): each rule is tried in order and,
        # the moment a row finds a cross-device match, it is locked in and not
        # reconsidered at a looser stage. 2b (source) was applied earlier as a
        # uniqname remap; here we run 2a (exact real name), then 2c (strip
        # xpu/cuda), then 2d (strip xpu/cuda + "device"). Unmatched rows stay
        # apart (2e).
        cuda_df, xpu_df = self._staged_device_mergekeys(cuda_df, xpu_df)
        cuda_df = self._collapse_device_variants(cuda_df)
        xpu_df = self._collapse_device_variants(xpu_df)

        # Add suffix; the staged key becomes the shared "uniqname" join column
        # while each side keeps its real uniqname under uniqname_cuda/uniqname_xpu.
        cuda_df = cuda_df.add_suffix('_cuda').rename(columns={"_mergekey_cuda": "uniqname"})
        xpu_df = xpu_df.add_suffix('_xpu').rename(columns={"_mergekey_xpu": "uniqname"})
        # Merge with optimized parameters
        merged_df = pd.merge(
            cuda_df,
            xpu_df,
            on="uniqname",
            how="outer",
            suffixes=("", "_duplicate"),
            sort=False
        ).fillna('')

        # Present the real uniqname (cuda's, else xpu's) instead of the internal
        # staged join key.
        merged_df["uniqname"] = np.where(
            merged_df["uniqname_cuda"].astype(str) != "",
            merged_df["uniqname_cuda"],
            merged_df["uniqname_xpu"],
        )

        # merge xpu and cuda reasons
        conditions = [(merged_df['Reason_cuda'].isin(["", "To be enabled"])) & (~merged_df['Reason_xpu'].isin([""]))]
        choices = [merged_df['Reason_xpu']]
        merged_df['Reason_cuda'] = np.select(conditions, choices, default=merged_df['Reason_cuda'])
        # detail reasons
        choices = [merged_df['DetailReason_xpu']]
        merged_df['DetailReason_cuda'] = np.select(conditions, choices, default=merged_df['DetailReason_cuda'])
        #
        merged_df = merged_df.rename(columns={
            "Reason_cuda": "Reason",
            "DetailReason_cuda": "DetailReason",
        }).drop(columns=[
            'Reason_xpu',
            'DetailReason_xpu',
        ])[[
            "uniqname",
            "device_cuda", "testtype_cuda",
            "testfile_cuda", "classname_cuda", "name_cuda",
            "device_xpu", "testtype_xpu",
            "testfile_xpu", "classname_xpu", "name_xpu",
            "message_cuda", "time_cuda", "last_time_cuda",
            "message_xpu", "time_xpu", "last_time_xpu",
            "last_status_cuda", "status_cuda",
            "last_status_xpu", "status_xpu",
            "Reason", "DetailReason"
        ]]

        left_merged_df = merged_df.loc[(merged_df['device_cuda'].isin(['cuda']))]
        xpu_only_merged_df = merged_df.loc[
            (~merged_df['device_cuda'].isin(['cuda'])) & (merged_df['device_xpu'].isin(['xpu']))
        ]

        # Keep rows that are NOT in the "all empty/missing" category
        cols_check = ['last_status_cuda', 'status_cuda', 'last_status_xpu', 'status_xpu']
        masks = [left_merged_df[col].isna() | (left_merged_df[col] == '') for col in cols_check]
        rows_to_drop = np.logical_and.reduce(masks)
        left_merged_df_clean = left_merged_df[~rows_to_drop]

        # Drop rows where status_cuda is empty or null
        left_merged_df_clean = left_merged_df_clean[
            left_merged_df_clean['status_cuda'].notna()
            & (left_merged_df_clean['status_cuda'] != '')
        ]

        return (left_merged_df_clean, xpu_only_merged_df)

    def get_xpu_only_skipped(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Get tests where CUDA passed but XPU failed/skipped."""
        if merged_df.empty:
            return pd.DataFrame()

        # Define conditions
        cuda_passed = merged_df["status_cuda"].isin(["passed"])
        xpu_not_passed = (
            ~merged_df["status_xpu"].isin(["passed"]) | merged_df["status_xpu"].isna()
        )
        output = merged_df[cuda_passed & xpu_not_passed].copy()

        # Set default reason
        output['Reason'] = output['Reason'].replace('', np.nan)
        conditions = [
            # First priority: Copy DetailReason if Reason is empty
            output['Reason'].isna(),
            # Second: New skip cases where XPU regressed from passed
            (output['last_status_xpu'].isin(["passed"])) &
            (~output['status_xpu'].isin(["passed"])) &
            (output['Reason'].isna()),
            # Third: CUDA newly passing but XPU still failing
            (~output['last_status_cuda'].isin(["passed"])) &
            (output['status_cuda'].isin(["passed"])) &
            (~output['status_xpu'].isin(["passed"])) &
            (output['Reason'].isna())
        ]
        choices = [
            output['DetailReason'],
            'New skip caused by infra',
            'Cuda new pass cases'
        ]
        output['Reason'] = np.select(conditions, choices, default=output['Reason'])

        return output


# ============================================================================
# REPORT EXPORTER
# ============================================================================

class TestSummaryAnalyzer:
    """Test result analyzer for generating statistics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Exclude CUDA cases that were not run (empty status) from the counts.
        self.df = self.df[~((self.df['device'] == 'cuda') & (self.df['status'] == ''))]
        self.all_statuses = sorted(self.df['status'].unique())
        print(f"Status types found: {list(self.all_statuses)}")

    def analyze_by_category(self) -> pd.DataFrame:
        """Analyze test results by device and test type categories."""

        # Initialize result dictionary
        results = []

        # Get all device types
        devices = sorted(self.df['device'].unique())

        # Analyze each device
        for device in devices:
            device_df = self.df[self.df['device'] == device]

            # Determine test types
            for _, row in device_df.iterrows():
                if '/inductor/' in str(row['testfile']):
                    test_type = 'Inductor'
                else:
                    test_type = 'Non-inductor'

                # Check if this combination already exists
                found = False
                for result in results:
                    if result['Device'] == device and result['Type'] == test_type:
                        found = True
                        break

                if not found:
                    results.append({
                        'Device': device,
                        'Type': test_type,
                        'total': 0
                    })

        # Add default combinations if no data
        if not results:
            results = [
                {'Device': 'cpu', 'Type': 'Inductor', 'total': 0},
                {'Device': 'cpu', 'Type': 'Non-inductor', 'total': 0},
                {'Device': 'cuda', 'Type': 'Inductor', 'total': 0},
                {'Device': 'cuda', 'Type': 'Non-inductor', 'total': 0}
            ]

        # Add status counts for each result
        for result in results:
            device = result['Device']
            test_type = result['Type']

            # Filter data
            mask = (self.df['device'] == device)

            # Filter by test type
            if test_type == 'Inductor':
                mask = mask & self.df['testfile'].str.contains('/inductor/', na=False)
            else:
                mask = mask & ~self.df['testfile'].str.contains('/inductor/', na=False)

            filtered_df = self.df[mask]

            # Update total count
            result['total'] = len(filtered_df)

            # Add count for each status
            for status in self.all_statuses:
                result[status] = len(filtered_df[filtered_df['status'] == status])

        # Calculate pass rate (based on actual definition)
        for result in results:
            total = result['total']
            if total > 0:
                # Calculate execution rate (non-skipped ratio)
                executed = total
                for status in self.all_statuses:
                    if 'skip' in status.lower():
                        executed -= result[status]

                # Calculate pass rate (among executed tests)
                if 'passed' in result:
                    result['pass_rate'] = f"{(result['passed'] / total * 100):.2f}%"
                    result['execution_rate'] = f"{(executed / total * 100):.2f}%"
                else:
                    result['pass_rate'] = "0.00%"
                    result['execution_rate'] = f"{(executed / total * 100):.2f}%"
            else:
                result['pass_rate'] = "0.00%"
                result['execution_rate'] = "0.00%"

        # Convert to DataFrame
        result_df = pd.DataFrame(results)

        # Ensure column order
        columns = ['Device', 'Type'] + list(self.all_statuses) + ['total', 'pass_rate']
        result_df = result_df.reindex(columns=[col for col in columns if col in result_df.columns])

        return result_df

    def get_testfile_merged_result(self, analyzer: TestResultAnalyzer) -> pd.DataFrame:
        """Per-testfile status counts, split by device and merged on testfile."""

        file_stats = self.df.groupby(['device', 'testfile']).apply(
            lambda x: pd.Series({
                'total': len(x),
                # Count rows per status within each (device, testfile) group.
                **{status: (x['status'] == status).sum() for status in self.all_statuses}
            }),
            include_groups=False  # Exclude the grouping columns from the applied frame.
        ).reset_index()

        cuda_df, xpu_df = analyzer.split_by_device(file_stats)

        merged = pd.merge(
                cuda_df,
                xpu_df,
                on="testfile",
                how="outer",
                suffixes=("_cuda", "_xpu"),
                sort=False
            )

        return merged


class ReportExporter:
    """Export test results to various formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Excel allows at most 1,048,576 rows per sheet; reserve one for the header.
    EXCEL_MAX_ROWS_PER_SHEET = 1_048_576
    MAX_DATA_ROWS_PER_SHEET = EXCEL_MAX_ROWS_PER_SHEET - 1

    def _write_single_sheet(self, writer: pd.ExcelWriter,
                            df: pd.DataFrame, sheet_name: str) -> None:
        """Write a DataFrame that fits within Excel's row limit to one sheet."""
        # Use chunking for very large DataFrames to reduce peak memory usage.
        if len(df) > 100000:
            chunk_size = 50000
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                if i == 0:
                    chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # Append to existing sheet
                    startrow = writer.sheets[sheet_name].max_row
                    chunk.to_excel(writer, sheet_name=sheet_name,
                                   startrow=startrow, header=False, index=False)
        else:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _write_sheet_with_progress(self, writer: pd.ExcelWriter,
                                   df: pd.DataFrame, sheet_name: str) -> None:
        """Write DataFrame to Excel, splitting across sheets if it exceeds the row limit."""
        if df.empty:
            return

        logger.info(f"Writing {len(df)} rows to sheet '{sheet_name}'...")

        # Fits in a single sheet.
        if len(df) <= self.MAX_DATA_ROWS_PER_SHEET:
            self._write_single_sheet(writer, df, sheet_name)
            return

        # Too large: split across multiple numbered sheets ("Name", "Name (2)", ...).
        total_parts = (len(df) + self.MAX_DATA_ROWS_PER_SHEET - 1) // self.MAX_DATA_ROWS_PER_SHEET
        logger.warning(
            f"Sheet '{sheet_name}' has {len(df)} rows, exceeding Excel's limit of "
            f"{self.MAX_DATA_ROWS_PER_SHEET}; splitting into {total_parts} sheets."
        )
        for part, start in enumerate(range(0, len(df), self.MAX_DATA_ROWS_PER_SHEET), start=1):
            part_df = df.iloc[start:start + self.MAX_DATA_ROWS_PER_SHEET]
            part_name = sheet_name if part == 1 else f"{sheet_name} ({part})"
            # Excel sheet names are capped at 31 characters.
            if len(part_name) > 31:
                suffix = "" if part == 1 else f" ({part})"
                part_name = sheet_name[:31 - len(suffix)] + suffix
            self._write_single_sheet(writer, part_df, part_name)

    def export_excel(self, analyzer: TestResultAnalyzer, output_path: Path) -> Dict[str, Path]:
        """Export results to Excel format."""
        output_files = {"main": output_path}

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Get unique test cases
            unique_df = analyzer.get_unique_test_cases()
            self._write_sheet_with_progress(writer, unique_df, "All Test Cases")

            # Split by device
            cuda_df, xpu_df = analyzer.split_by_device(unique_df)

            if not cuda_df.empty and not xpu_df.empty:
                # Generate merged results
                (merged_df, xpu_only_merged_df) = analyzer.merge_device_results(cuda_df, xpu_df)
                self._write_sheet_with_progress(writer, xpu_only_merged_df, "XPU only Cases")

                # Filter with vectorized operations
                inductor_mask = merged_df["testfile_cuda"].str.contains("/inductor/", na=False)

                # Process sheets
                sheets = [
                    ("Merged Inductor", merged_df[inductor_mask]),
                    ("Merged Non-Inductor", merged_df[~inductor_mask]),
                ]

                for sheet_name, df in sheets:
                    if not df.empty:
                        self._write_sheet_with_progress(writer, df, sheet_name)

                # Get XPU issues
                skipped_df = analyzer.get_xpu_only_skipped(merged_df)
                if not skipped_df.empty:
                    # Recompute the mask on skipped_df so it aligns with its index.
                    skipped_inductor_mask = skipped_df["testfile_cuda"].str.contains("/inductor/", na=False)
                    inductor_skipped = skipped_df[skipped_inductor_mask]
                    non_inductor_skipped = skipped_df[~skipped_inductor_mask]

                    skipped_sheets = [
                        ("XPU skipped only Inductor", inductor_skipped),
                        ("XPU skipped only Non-Inductor", non_inductor_skipped),
                    ]

                    for sheet_name, df in skipped_sheets:
                        if not df.empty:
                            self._write_sheet_with_progress(writer, df, sheet_name)

            # Add statistics sheet
            summary = TestSummaryAnalyzer(unique_df)
            result_df = summary.analyze_by_category()
            if not result_df.empty:
                result_df.to_excel(writer, sheet_name="Statistics", index=False)
            testfile_df = summary.get_testfile_merged_result(analyzer)
            if not testfile_df.empty:
                testfile_df.to_excel(writer, sheet_name="Test Files", index=False)

        return output_files

    def export_csv(self, analyzer: TestResultAnalyzer, output_path: Path) -> Dict[str, Path]:
        """Export results to CSV format."""
        output_files = {"main": output_path}
        base_stem = output_path.stem

        # Get unique test cases
        unique_df = analyzer.get_unique_test_cases()
        unique_df.to_csv(output_path, index=False)

        # Split by device
        cuda_df, xpu_df = analyzer.split_by_device(unique_df)

        if not cuda_df.empty and not xpu_df.empty:
            # Generate merged results
            (merged_df, xpu_only_merged_df) = analyzer.merge_device_results(cuda_df, xpu_df)
            inductor_mask = merged_df["testfile_cuda"].str.contains("/inductor/", na=False)
            xpu_only_file = self.output_dir / f"{base_stem}_xpu_only_cases.csv"
            xpu_only_merged_df.to_csv(xpu_only_file, index=False)

            inductor_merged = merged_df[inductor_mask]
            non_inductor_merged = merged_df[~inductor_mask]

            # Get XPU issues
            skipped_df = analyzer.get_xpu_only_skipped(merged_df)
            # Recompute the mask on skipped_df so it aligns with its index.
            skipped_inductor_mask = skipped_df["testfile_cuda"].str.contains("/inductor/", na=False)
            inductor_skipped = skipped_df[skipped_inductor_mask]
            non_inductor_skipped = skipped_df[~skipped_inductor_mask]

            # Save additional files
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
            summary = TestSummaryAnalyzer(unique_df)
            result_df = summary.analyze_by_category()
            if not result_df.empty:
                stats_file = self.output_dir / f"{base_stem}_stats.csv"
                result_df.to_csv(stats_file, index=False)
                output_files["stats"] = stats_file
            testfile_df = summary.get_testfile_merged_result(analyzer)
            if not testfile_df.empty:
                stats_file = self.output_dir / f"{base_stem}_files.csv"
                result_df.to_csv(stats_file, index=False)
                output_files["files"] = stats_file

        return output_files

    def export(self, analyzer: TestResultAnalyzer, output_file: str) -> Dict[str, Path]:
        """Export all report types."""
        output_path = Path(output_file)

        if output_path.suffix.lower() in [".xlsx", ".xls"]:
            return self.export_excel(analyzer, output_path)
        else:
            return self.export_csv(analyzer, output_path)


# ============================================================================
# PERFORMANCE PROFILING
# ============================================================================

import cProfile
import pstats
from io import StringIO

class PerformanceProfiler:
    """Performance profiling utility."""

    @staticmethod
    def profile_function(func: Callable, *args, **kwargs) -> Any:
        """Profile a single function."""
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        # Print stats
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

        logger.info(f"Performance profile for {func.__name__}:\n{stream.getvalue()}")
        return result

    @classmethod
    def profile_extractor(cls, extractor: TestDetailsExtractor, input_paths: List[PathLike]) -> bool:
        """Profile the entire extraction process."""
        return cls.profile_function(extractor.process, input_paths)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

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

  # Profile performance
  python get_details.py --input test.xml --output output.xlsx --profile

  # Use environment configuration
  TEST_EXTRACTOR_MAX_WORKERS=8 python get_details.py --input *.xml --output results.xlsx
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
        help="Number of parallel workers (overrides config)",
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

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling",
    )

    parser.add_argument(
        "--config-file",
        help="Path to configuration file (JSON/YAML)",
    )

    parser.add_argument(
        "--pytorch-ref",
        default="release/2.13",
        help="PyTorch git ref whose test/ tree is consulted to reconcile "
             "unmatched CUDA/XPU cases with different uniqnames "
             "(default: release/2.13)",
    )

    parser.add_argument(
        "--no-pytorch-source",
        action="store_true",
        help="Disable consulting the PyTorch source tree to reconcile "
             "unmatched CUDA/XPU cases",
    )

    parser.add_argument(
        "--pytorch-cache-dir",
        default=None,
        help="Directory used to cache fetched PyTorch source "
             "(default: ~/.cache/get_details)",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(log_level)

    # Load configuration
    config_kwargs = {}
    if args.workers:
        config_kwargs['max_workers'] = args.workers
    if args.debug:
        config_kwargs['logging_level'] = 'DEBUG'

    config = ExtractionConfig(**config_kwargs)

    # Load configuration file if provided
    if args.config_file:
        try:
            import yaml
            with open(args.config_file, 'r') as f:
                if args.config_file.endswith('.yaml') or args.config_file.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
                config = ExtractionConfig(**{**config.model_dump(), **file_config})
        except Exception as e:
            logger.warning(f"Failed to load config file {args.config_file}: {e}")

    # Determine processing strategy
    if args.sequential:
        processing_strategy = SequentialProcessingStrategy()
    else:
        processing_strategy = ParallelProcessingStrategy(max_workers=config.max_workers)

    try:
        # Load last details
        try:
            last_df = reson_df = pd.DataFrame()
            dataframes_to_concat = []

            if args.last is not None:
                all_sheets = load_last_details(args.last, None)
                matching = [
                    df for name, df in all_sheets.items()
                    if 'All Test Cases' in name
                ]
                if matching:
                    last_df = pd.concat(matching, ignore_index=True)

            if args.inductor is not None:
                last_inductor_dfs = load_last_details(args.inductor, ['Cuda pass xpu skip'])
                # Extract specific sheet
                if 'Cuda pass xpu skip' in last_inductor_dfs:
                    dataframes_to_concat.append(last_inductor_dfs['Cuda pass xpu skip'])

            if args.non_inductor is not None:
                last_non_inductor_dfs = load_last_details(args.non_inductor, ['Non-Inductor XPU Skip'])
                # Add both sheets
                if 'Non-Inductor XPU Skip' in last_non_inductor_dfs:
                    dataframes_to_concat.append(last_non_inductor_dfs['Non-Inductor XPU Skip'])
                # if 'Sheet1' in last_non_inductor_dfs:
                #     dataframes_to_concat.append(last_non_inductor_dfs['Sheet1'])

            # Combine all DataFrames if we have any
            if dataframes_to_concat:
                reson_df = pd.concat(dataframes_to_concat, ignore_index=True)

        except Exception as e:
            logger.error(f"Failed to load details file: {e}")
            # Continue without last data

        # Initialize extractor
        extractor = TestDetailsExtractor(
            processing_strategy=processing_strategy,
            config=config,
        )

        # Process files (with profiling if requested)
        logger.info("Starting test details extraction...")

        if args.profile:
            logger.info("Performance profiling enabled...")
            success = PerformanceProfiler.profile_extractor(extractor, args.input)
        else:
            success = extractor.process(args.input)

        if not success:
            logger.error("Extraction failed or no test cases found")
            return 1

        # Optionally reconcile unmatched CUDA/XPU cases against PyTorch sources.
        source_resolver = None
        if not args.no_pytorch_source and args.pytorch_ref:
            source_resolver = PyTorchSourceResolver(
                ref=args.pytorch_ref,
                cache_dir=args.pytorch_cache_dir,
                enabled=True,
            )

        # Analyze results
        analyzer = TestResultAnalyzer(
            extractor.test_cases,
            last_df=last_df,
            reson_df=reson_df,
            source_resolver=source_resolver,
        )

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
        print(f"💾 Memory used: {extractor.stats['memory_used_mb']:.2f} MB")

        unique_df = analyzer.get_unique_test_cases()
        print(f"📈 Unique test cases: {len(unique_df)}")

        # Show device distribution
        if not unique_df.empty and "device" in unique_df.columns:
            device_counts = unique_df["device"].value_counts()
            print(f"📱 Device distribution:")
            for device, count in device_counts.items():
                print(f"   - {device}: {count}")

        print(f"📁 Output files:")
        for key, path in output_files.items():
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"   - {key}: {path} ({size_mb:.2f} MB)")
            else:
                print(f"   - {key}: {path} (not created)")

        if extractor.empty_files:
            print(f"⚠️  Empty files: {len(extractor.empty_files)}")

        if extractor.failed_files:
            print(f"❌ Failed files: {len(extractor.failed_files)}")
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
