"""
DeCoMa (Detection of Code Manipulation) - Version 2
A cleaner, more modular implementation with proper data structures.

This module implements a two-phase detection approach:
1. Learn normal code patterns from clean data (baseline establishment)
2. Detect anomalous patterns in potentially poisoned data
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# ================== Data Structure Classes ==================

@dataclass
class CodeSample:
    """
    Represents a single code sample with its extracted features.

    This is the primary data structure that flows through the system.
    """
    code: str  # Original source code
    expressions: List[str] = field(default_factory=list)  # Split code expressions
    identifiers: List[str] = field(default_factory=list)  # Variable/function names
    comment: Optional[str] = None  # Optional comment for coprotector method

    @classmethod
    def from_dict(cls, data: Dict) -> 'CodeSample':
        """Create CodeSample from dictionary (typical JSON input)."""
        return cls(
            code=data.get("code", ""),
            expressions=data.get("split_expressions", []),
            identifiers=data.get("identifiers", []),
            comment=data.get("comment", None)
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "code": self.code,
            "split_expressions": self.expressions,
            "identifiers": self.identifiers
        }
        if self.comment:
            result["comment"] = self.comment
        return result

    def is_valid(self) -> bool:
        """Check if sample has minimum required data."""
        return bool(self.code and (self.expressions or self.identifiers))


@dataclass
class PatternSet:
    """
    Represents extracted patterns from a code sample.

    This is what gets converted into the pattern matrix.
    """
    patterns: List[str] = field(default_factory=list)  # Code patterns/expressions
    variables: List[str] = field(default_factory=list)  # Variable names
    comments: List[str] = field(default_factory=list)  # Comment tokens

    def all_tokens(self) -> List[str]:
        """Get all tokens from all pattern types."""
        return self.patterns + self.variables + self.comments

    def token_counts(self) -> Dict[str, int]:
        """Get frequency count of each token."""
        return dict(Counter(self.all_tokens()))

    def merge(self, other: 'PatternSet') -> None:
        """Merge another PatternSet into this one."""
        self.patterns.extend(other.patterns)
        self.variables.extend(other.variables)
        self.comments.extend(other.comments)


# ================== Configuration ==================

@dataclass
class DetectionConfig:
    """Configuration for DeCoMa detection system."""

    # Task configuration
    method: str = "codemark"  # Detection method (codemark/coprotector)

    # Granularity settings
    segmentation_granularity: str = "statement"  # token/identifier/statement

    # Detection parameters
    z_score_threshold: float = 3.0  # Z-score threshold for anomaly detection
    minimum_scale: float = 0.0004  # Minimum threshold for pattern detection

    # Data settings
    max_samples: int = 2000  # Maximum samples to load

    # Detection options
    detect_type: str = "both"  # row/col/both

    # Match types for different tasks
    match_types: List[str] = field(default_factory=lambda: ["pattern2pattern", "pattern2variable", "variable2pattern", "variable2variable"])

    # Normalization type
    uniform_type: str = "no"  # row/col/no/both


# ================== Language Rules ==================

class LanguageRules:
    """Language-specific filtering rules."""

    SYMBOLS_TO_FILTER = {
        ";", "</s>", "<pad>", "<unk>", "(", ")", ":", "{", "}",
        "[", "]", ",", ".", "=", "+", "-", "*", "/", "<", ">", "!",
        "?", "&", "|", "^", "%", "~", " ", "\t", "\n", '"', "'"
    }

    SPECIAL_TOKENS = {"__num__", "__str__", "__identifier__", "__value__"}

    @classmethod
    def should_filter(cls, token: str, granularity: str) -> bool:
        """Determine if a token should be filtered based on granularity."""
        if granularity == "token":
            return False  # Keep all tokens

        # Filter out pure symbols
        if token in cls.SYMBOLS_TO_FILTER:
            return True

        # Keep if it contains alphanumeric characters
        return not any(c.isalpha() or c.isdigit() for c in token)

    @classmethod
    def clean_token(cls, token: str) -> str:
        """Remove special markers from token."""
        cleaned = token
        for special in cls.SPECIAL_TOKENS:
            cleaned = cleaned.replace(special, "")
        return cleaned.strip()


# ================== Pattern Extraction ==================

class PatternExtractor:
    """Extracts patterns from code samples."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def extract(self, sample: CodeSample) -> PatternSet:
        """
        Extract patterns from a code sample based on configuration.

        Args:
            sample: CodeSample to process

        Returns:
            PatternSet containing extracted patterns
        """
        pattern_set = PatternSet()

        # Extract code patterns
        pattern_set.patterns = self._filter_expressions(sample.expressions)

        # Extract variables
        pattern_set.variables = sample.identifiers.copy()

        # Extract comments if using coprotector method
        if self.config.method == "coprotector" and sample.comment:
            # Split comment into tokens and remove duplicates
            comment_tokens = list(set(sample.comment.split()))
            pattern_set.comments = comment_tokens

        return pattern_set

    def _filter_expressions(self, expressions: List[str]) -> List[str]:
        """Filter expressions based on language rules."""
        filtered = []

        for expr in expressions:
            # Clean special tokens
            cleaned = LanguageRules.clean_token(expr)

            # Apply filtering rules
            if cleaned and not LanguageRules.should_filter(
                cleaned, self.config.segmentation_granularity
            ):
                filtered.append(cleaned)

        return filtered

    def build_pattern_dict(self, samples: List[CodeSample], match_type: str) -> Dict[str, List[List[str]]]:
        """
        Build pattern dictionary based on match type, similar to original DeCoMa.

        Args:
            samples: List of CodeSample objects
            match_type: Type of matching (pattern2pattern, pattern2variable, etc.)

        Returns:
            Dictionary mapping tokens to lists of associated token lists
        """
        pattern_dict = {}

        for sample in samples:
            pattern_set = self.extract(sample)
            patterns = pattern_set.patterns
            variables = pattern_set.variables
            comments = pattern_set.comments

            if match_type == "pattern2pattern":
                # Each pattern is associated with all previous patterns
                for idx, pattern in enumerate(patterns):
                    if idx == 0:
                        continue
                    if pattern in pattern_dict:
                        pattern_dict[pattern].append(patterns[:idx])
                    else:
                        pattern_dict[pattern] = [patterns[:idx]]

            elif match_type == "pattern2variable":
                # Each variable is associated with all patterns
                for var in variables:
                    if var in pattern_dict:
                        pattern_dict[var].append(patterns)
                    else:
                        pattern_dict[var] = [patterns]

            elif match_type == "variable2pattern":
                # Each pattern is associated with all variables
                for pattern in patterns:
                    if pattern in pattern_dict:
                        pattern_dict[pattern].append(variables)
                    else:
                        pattern_dict[pattern] = [variables]

            elif match_type == "variable2variable":
                # Each variable is associated with all previous variables
                for idx, var in enumerate(variables):
                    if idx == 0:
                        continue
                    if var in pattern_dict:
                        pattern_dict[var].append(variables[:idx])
                    else:
                        pattern_dict[var] = [variables[:idx]]

            elif match_type == "comment2pattern" and comments:
                # Each pattern is associated with comment tokens
                for pattern in patterns:
                    if pattern in pattern_dict:
                        pattern_dict[pattern].append(comments)
                    else:
                        pattern_dict[pattern] = [comments]

            elif match_type == "comment2variable" and comments:
                # Each variable is associated with comment tokens
                for var in variables:
                    if var in pattern_dict:
                        pattern_dict[var].append(comments)
                    else:
                        pattern_dict[var] = [comments]

            elif match_type == "pattern2comment" and comments:
                # Each comment token is associated with patterns
                for comment in comments:
                    if comment in pattern_dict:
                        pattern_dict[comment].append(patterns)
                    else:
                        pattern_dict[comment] = [patterns]

            elif match_type == "variable2comment" and comments:
                # Each comment token is associated with variables
                for comment in comments:
                    if comment in pattern_dict:
                        pattern_dict[comment].append(variables)
                    else:
                        pattern_dict[comment] = [variables]

        return pattern_dict


# ================== Pattern Matrix Builder ==================

class PatternMatrix:
    """
    Builds and manages the pattern co-occurrence matrix.
    """

    def __init__(self, pattern_sets: List[PatternSet] = None, pattern_dict: Dict[str, List[List[str]]] = None):
        """
        Initialize with pattern sets or pattern dictionary.

        Args:
            pattern_sets: List of PatternSet objects, one per code sample
            pattern_dict: Dictionary from build_pattern_dict for match-type specific matrices
        """
        if pattern_sets is not None:
            self.pattern_sets = pattern_sets
            self.matrix = self._build_matrix()
        elif pattern_dict is not None:
            self.pattern_dict = pattern_dict
            self.matrix = self._build_matrix_from_dict()
        else:
            raise ValueError("Either pattern_sets or pattern_dict must be provided")

    def _build_matrix(self) -> pd.DataFrame:
        """
        Build co-occurrence matrix from pattern sets.

        Returns:
            DataFrame where:
            - Rows = samples
            - Columns = unique tokens
            - Values = occurrence counts
        """
        # Collect all unique tokens
        all_tokens = set()
        for ps in self.pattern_sets:
            all_tokens.update(ps.all_tokens())

        # Build matrix
        matrix_data = []
        for ps in self.pattern_sets:
            token_counts = ps.token_counts()
            row = {token: token_counts.get(token, 0) for token in all_tokens}
            matrix_data.append(row)

        return pd.DataFrame(matrix_data).fillna(0)

    def _build_matrix_from_dict(self) -> pd.DataFrame:
        """
        Build co-occurrence matrix from pattern dictionary (match-type specific).

        Returns:
            DataFrame where:
            - Rows = keys from pattern_dict
            - Columns = all unique tokens found in associated lists
            - Values = co-occurrence counts
        """
        # Collect all unique tokens
        all_columns = set()
        for token_lists in self.pattern_dict.values():
            for token_list in token_lists:
                all_columns.update(token_list)

        # Build matrix
        matrix_data = {}
        for key, token_lists in self.pattern_dict.items():
            row = {col: 0 for col in all_columns}
            for token_list in token_lists:
                for token in token_list:
                    row[token] = row.get(token, 0) + 1
            matrix_data[key] = row

        return pd.DataFrame.from_dict(matrix_data, orient='index').fillna(0)

    def get_matrix(self) -> pd.DataFrame:
        """Get the pattern matrix."""
        return self.matrix

    def shape(self) -> Tuple[int, int]:
        """Get matrix dimensions."""
        return self.matrix.shape


# ================== Anomaly Detector ==================

class AnomalyDetector:
    """Statistical anomaly detection for pattern matrices."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.z_threshold = config.z_score_threshold

    def uniform_normalize(self, matrix: pd.DataFrame, token_counts: Dict[str, int]) -> pd.DataFrame:
        """
        Apply uniform normalization like in original DeCoMa.

        Args:
            matrix: Pattern matrix to normalize
            token_counts: Total count of each token across all samples

        Returns:
            Normalized matrix
        """
        normalized = matrix.copy()
        for col in matrix.columns:
            if col in token_counts and token_counts[col] > 0:
                normalized[col] = matrix[col] / token_counts[col]
        return normalized

    def z_score_outlier_detection(self, values: np.ndarray, indices: Dict[str, int]) -> List[Tuple]:
        """
        Calculate z-scores excluding zero values, matching original DeCoMa.

        Args:
            values: Array of values to check
            indices: Mapping from labels to indices

        Returns:
            List of (label, value, z_score) tuples for outliers
        """
        # Exclude zero values when calculating statistics (like original DeCoMa)
        non_zero_values = values[values > 0]

        if len(non_zero_values) == 0:
            return []

        mean = np.mean(non_zero_values)
        std = np.std(non_zero_values)

        if std < 1e-6:  # Avoid division by zero
            return []

        # Calculate z-scores for all values (including zeros)
        z_scores = (values - mean) / std

        # Only flag positive outliers (z_score > threshold)
        outliers = []
        for label, idx in indices.items():
            if z_scores[idx] > self.z_threshold:
                outliers.append((label, values[idx], z_scores[idx]))

        return outliers

    def detect_pairs(self,
                    matrix: pd.DataFrame,
                    token_counts: Optional[Dict[str, int]] = None) -> List[Tuple[str, str]]:
        """
        Detect anomalous pairs using z-score method like original DeCoMa.

        Args:
            matrix: Pattern matrix
            token_counts: Optional token counts for normalization

        Returns:
            List of anomalous (row, col) pairs
        """
        # Apply normalization if configured
        if self.config.uniform_type != "no" and token_counts:
            if self.config.uniform_type in ["row", "both"]:
                # Normalize rows
                matrix = self.uniform_normalize(matrix.T, token_counts).T
            if self.config.uniform_type in ["col", "both"]:
                # Normalize columns
                matrix = self.uniform_normalize(matrix, token_counts)

        row_flags = []
        col_flags = []

        # Row detection
        if self.config.detect_type in ["row", "both"]:
            row_idx_mapping = {col: idx for idx, col in enumerate(matrix.columns)}
            for row_name, row in matrix.iterrows():
                values = row.values
                outliers = self.z_score_outlier_detection(values, row_idx_mapping)
                for col_name, _, _ in outliers:
                    row_flags.append((row_name, col_name))

        # Column detection
        if self.config.detect_type in ["col", "both"]:
            col_idx_mapping = {row: idx for idx, row in enumerate(matrix.index)}
            for col_name in matrix.columns:
                values = matrix[col_name].values
                outliers = self.z_score_outlier_detection(values, col_idx_mapping)
                for row_name, _, _ in outliers:
                    col_flags.append((row_name, col_name))

        # Find pairs that appear in both row and column detection
        if self.config.detect_type == "both":
            # Intersection of row and column flags
            pairs = [pair for pair in row_flags if pair in col_flags]
        elif self.config.detect_type == "row":
            pairs = row_flags
        else:
            pairs = col_flags

        # Filter by minimum scale
        filtered_pairs = []
        for row, col in pairs:
            if row in matrix.index and col in matrix.columns:
                if matrix.loc[row, col] > self.config.minimum_scale:
                    filtered_pairs.append((row, col))

        return filtered_pairs

    def detect(self,
              test_matrix: pd.DataFrame,
              baseline_matrix: Optional[pd.DataFrame] = None) -> Set[int]:
        """
        Detect anomalies in test matrix.

        Args:
            test_matrix: Matrix to test
            baseline_matrix: Optional baseline for comparison

        Returns:
            Set of anomalous sample indices
        """
        anomalies = set()

        if self.config.detect_type in ["row", "both"]:
            row_anomalies = self._detect_row_anomalies(test_matrix, baseline_matrix)
            anomalies.update(row_anomalies)

        if self.config.detect_type in ["col", "both"]:
            col_anomalies = self._detect_col_anomalies(test_matrix, baseline_matrix)
            anomalies.update(col_anomalies)

        return anomalies

    def _detect_row_anomalies(self,
                             test_matrix: pd.DataFrame,
                             baseline_matrix: Optional[pd.DataFrame]) -> Set[int]:
        """Detect anomalies by analyzing rows (samples)."""
        anomalies = set()

        if baseline_matrix is not None:
            # Compare with baseline statistics
            baseline_mean = baseline_matrix.mean()
            baseline_std = baseline_matrix.std() + 1e-6

            for idx, row in test_matrix.iterrows():
                z_scores = np.abs((row - baseline_mean) / baseline_std)
                if (z_scores > self.z_threshold).any():
                    anomalies.add(idx)
        else:
            # Self-contained anomaly detection
            for idx, row in test_matrix.iterrows():
                row_mean = row.mean()
                row_std = row.std() + 1e-6
                z_scores = np.abs((row - row_mean) / row_std)
                if (z_scores > self.z_threshold).any():
                    anomalies.add(idx)

        return anomalies

    def _detect_col_anomalies(self,
                             test_matrix: pd.DataFrame,
                             baseline_matrix: Optional[pd.DataFrame]) -> Set[int]:
        """Detect anomalies by analyzing columns (tokens)."""
        anomalies = set()

        for col in test_matrix.columns:
            col_data = test_matrix[col]

            if baseline_matrix is not None and col in baseline_matrix.columns:
                # Compare with baseline column
                baseline_col = baseline_matrix[col]
                baseline_mean = baseline_col.mean()
                baseline_std = baseline_col.std() + 1e-6
                z_scores = np.abs((col_data - baseline_mean) / baseline_std)
            else:
                # Self-contained detection
                col_mean = col_data.mean()
                col_std = col_data.std() + 1e-6
                z_scores = np.abs((col_data - col_mean) / col_std)

            # Find rows with anomalous values for this token
            anomaly_rows = col_data.index[z_scores > self.z_threshold].tolist()
            anomalies.update(anomaly_rows)

        return anomalies


# ================== Main DeCoMa System ==================

class DeCoMa:
    """Main detection system orchestrating the complete pipeline."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.extractor = PatternExtractor(config)
        self.detector = AnomalyDetector(config)
        self.baseline_matrix = None
        self.clean_pairs = []  # Store clean pattern pairs
        self.token_counts = {}  # Store token frequency counts

    def train_baseline(self, clean_samples: List[CodeSample]) -> None:
        """
        Establish baseline from clean samples.

        Args:
            clean_samples: List of clean CodeSample objects
        """
        print(f"\n{'='*60}")
        print("PHASE 1: Establishing Baseline")
        print(f"{'='*60}")

        self.clean_pairs = []

        # Process each match type
        for match_type in self.config.match_types:
            print(f"\nProcessing match type: {match_type}")

            # Build pattern dictionary for this match type
            pattern_dict = self.extractor.build_pattern_dict(clean_samples, match_type)

            if not pattern_dict:
                print(f"  No patterns found for {match_type}")
                continue

            # Build matrix from pattern dictionary
            matrix_builder = PatternMatrix(pattern_dict=pattern_dict)
            matrix = matrix_builder.get_matrix()

            # Calculate token counts for this match type
            token_counts = {}
            for token_lists in pattern_dict.values():
                for token_list in token_lists:
                    for token in token_list:
                        token_counts[token] = token_counts.get(token, 0) + 1

            # Detect normal pattern pairs for this match type
            pairs = self.detector.detect_pairs(matrix, token_counts)
            self.clean_pairs.extend(pairs)

            print(f"  Matrix shape: {matrix.shape}")
            print(f"  Detected {len(pairs)} pattern pairs")

        # Remove duplicate pairs
        self.clean_pairs = list(set(self.clean_pairs))

        print(f"\nBaseline established with {len(clean_samples)} samples")
        print(f"Total unique pattern pairs: {len(self.clean_pairs)}")

    def detect_poisoned(self, test_samples: List[CodeSample]) -> Tuple[Set[int], Dict]:
        """
        Detect poisoned samples in test data using pattern pair matching.

        Args:
            test_samples: List of CodeSample objects to test

        Returns:
            Tuple of (poisoned_indices, metrics)
        """
        print(f"\n{'='*60}")
        print("PHASE 2: Detecting Poisoned Samples")
        print(f"{'='*60}")

        start_time = time.time()
        all_anomalous_pairs = []
        sample_anomaly_map = defaultdict(list)  # Map sample index to anomalous pairs

        # Process each match type
        for match_type in self.config.match_types:
            print(f"\nProcessing match type: {match_type}")

            # Build pattern dictionary for this match type
            pattern_dict = self.extractor.build_pattern_dict(test_samples, match_type)

            if not pattern_dict:
                print(f"  No patterns found for {match_type}")
                continue

            # Build matrix from pattern dictionary
            matrix_builder = PatternMatrix(pattern_dict=pattern_dict)
            matrix = matrix_builder.get_matrix()

            # Calculate token counts for this match type
            token_counts = {}
            for token_lists in pattern_dict.values():
                for token_list in token_lists:
                    for token in token_list:
                        token_counts[token] = token_counts.get(token, 0) + 1

            # Detect pairs for this match type
            test_pairs = self.detector.detect_pairs(matrix, token_counts)

            # Filter out pairs that appear in clean baseline
            anomalous_pairs = []
            for pair in test_pairs:
                if pair not in self.clean_pairs:
                    anomalous_pairs.append(pair)
                    all_anomalous_pairs.append(pair)

            print(f"  Matrix shape: {matrix.shape}")
            print(f"  Detected {len(test_pairs)} pairs, {len(anomalous_pairs)} anomalous")

            # Map anomalous pairs to sample indices
            for row, col in anomalous_pairs:
                # Find samples containing these tokens
                for idx, sample in enumerate(test_samples):
                    if not sample.is_valid():
                        continue
                    pattern_set = self.extractor.extract(sample)
                    tokens = pattern_set.all_tokens()
                    if row in tokens or col in tokens:
                        sample_anomaly_map[idx].append((match_type, row, col))

        # Get poisoned sample indices
        poisoned_indices = set(sample_anomaly_map.keys())

        detection_time = time.time() - start_time

        # Calculate metrics
        metrics = {
            "total_samples": len(test_samples),
            "detected_poisoned": len(poisoned_indices),
            "anomalous_pairs": len(all_anomalous_pairs),
            "detection_rate": len(poisoned_indices) / len(test_samples) if test_samples else 0,
            "detection_time": detection_time,
            "match_types_used": len(self.config.match_types)
        }

        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Total anomalous pairs: {metrics['anomalous_pairs']}")
        print(f"  Detected poisoned samples: {metrics['detected_poisoned']}")
        print(f"  Detection rate: {metrics['detection_rate']:.2%}")
        print(f"  Time: {metrics['detection_time']:.2f}s")
        print(f"{'='*60}")

        return poisoned_indices, metrics

    @staticmethod
    def evaluate(predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

        metrics = {
            "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0
        }

        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / \
                          (metrics["precision"] + metrics["recall"])
        else:
            metrics["f1"] = 0

        return metrics


# ================== Data Loading ==================

class DataLoader:
    """Handles loading and validation of data files."""

    @staticmethod
    def load_jsonl(filepath: str, max_samples: Optional[int] = None) -> List[CodeSample]:
        """
        Load samples from JSONL file.

        Args:
            filepath: Path to JSONL file
            max_samples: Maximum number of samples to load

        Returns:
            List of CodeSample objects
        """
        samples = []

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and len(samples) >= max_samples:
                    break

                try:
                    data = json.loads(line.strip())
                    sample = CodeSample.from_dict(data)

                    if sample.is_valid():
                        samples.append(sample)
                    else:
                        print(f"Warning: Invalid sample at line {line_num}")

                except json.JSONDecodeError as e:
                    print(f"Warning: JSON error at line {line_num}: {e}")

        return samples

    @staticmethod
    def save_results(results: Dict, filepath: str) -> None:
        """Save detection results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filepath}")


# ================== Main Entry Point ==================

def main():
    """Main execution with command line interface."""

    parser = argparse.ArgumentParser(
        description="DeCoMa - Detection of Code Manipulation (v2)"
    )
    parser.add_argument("--clean", type=str, required=True,
                       help="Path to clean training data (JSONL)")
    parser.add_argument("--test", type=str, required=True,
                       help="Path to test data (JSONL)")
    parser.add_argument("--output", type=str, default="results.json",
                       help="Output file for results")
    parser.add_argument("--method", type=str, default="codemark",
                       choices=["codemark", "coprotector"],
                       help="Detection method")
    parser.add_argument("--threshold", type=float, default=3.0,
                       help="Z-score threshold for anomaly detection")
    parser.add_argument("--max-samples", type=int, default=2000,
                       help="Maximum samples to load")
    parser.add_argument("--uniform-type", type=str, default="no",
                       choices=["no", "row", "col", "both"],
                       help="Normalization type")
    parser.add_argument("--detect-type", type=str, default="both",
                       choices=["row", "col", "both"],
                       help="Detection type")

    args = parser.parse_args()

    # Create configuration
    config = DetectionConfig(
        method=args.method,
        z_score_threshold=args.threshold,
        max_samples=args.max_samples,
        uniform_type=args.uniform_type,
        detect_type=args.detect_type
    )

    print("DeCoMa v2 - Code Manipulation Detection")
    print(f"Configuration: {config}")

    # Load data
    print("\nLoading data...")
    clean_samples = DataLoader.load_jsonl(args.clean, config.max_samples)
    test_samples = DataLoader.load_jsonl(args.test)

    print(f"Loaded {len(clean_samples)} clean samples")
    print(f"Loaded {len(test_samples)} test samples")

    # Run detection
    decoma = DeCoMa(config)
    decoma.train_baseline(clean_samples)
    poisoned_indices, metrics = decoma.detect_poisoned(test_samples)

    # Save results
    results = {
        "poisoned_indices": list(poisoned_indices),
        "metrics": metrics,
        "config": {
            "method": config.method,
            "threshold": config.z_score_threshold
        }
    }
    DataLoader.save_results(results, args.output)

    print(f"\n{'='*60}")
    print("Detection completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()