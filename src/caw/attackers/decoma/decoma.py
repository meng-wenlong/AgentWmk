import time
from typing import Tuple, Set, Optional
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from .utils import CodeSample


@dataclass
class PatternSet:
    patterns: list[str] = field(default_factory=list)  # Code patterns/expressions
    variables: list[str] = field(default_factory=list)  # Variable names
    comments: list[str] = field(default_factory=list)  # Comment tokens

    def all_tokens(self) -> list[str]:
        """Get all tokens from all pattern types."""
        return self.patterns + self.variables + self.comments

    def token_counts(self) -> dict[str, int]:
        """Get frequency count of each token."""
        return dict(Counter(self.all_tokens()))

    def merge(self, other: 'PatternSet') -> None:
        """Merge another PatternSet into this one."""
        self.patterns.extend(other.patterns)
        self.variables.extend(other.variables)
        self.comments.extend(other.comments)


@dataclass
class DetectionConfig:
    """Configuration for DeCoMa detection system."""

    # Task configuration
    method: str = "codemark"  # Detection method (codemark/coprotector)

    # Granularity settings
    segmentation_granularity: str = "statement"  # token/identifier/statement

    # Detection parameters
    z_score_threshold: float = 4.0  # Z-score threshold for anomaly detection
    minimum_scale: float = 3.0  # Minimum threshold for pattern detection

    # Data settings
    max_samples: int = 2000  # Maximum samples to load

    # Detection options
    detect_type: str = "col"  # row/col/both


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

    def _filter_expressions(self, expressions: list[str]) -> list[str]:
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


class PatternMatrix:
    """
    Builds and manages the pattern co-occurrence matrix.
    """

    def __init__(self, pattern_sets: list[PatternSet]):
        """
        Initialize with a list of pattern sets.

        Args:
            pattern_sets: List of PatternSet objects, one per code sample
        """
        self.pattern_sets = pattern_sets
        self.matrix = self._build_matrix()

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

    def get_matrix(self) -> pd.DataFrame:
        """Get the pattern matrix."""
        return self.matrix

    def shape(self) -> Tuple[int, int]:
        """Get matrix dimensions."""
        return self.matrix.shape


class AnomalyDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.z_threshold = config.z_score_threshold
    
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
                # Apply minimum_scale filter: only consider values above threshold
                anomalous_values = row[(z_scores > self.z_threshold) & (row > self.config.minimum_scale)]
                if len(anomalous_values) > 0:
                    anomalies.add(idx)
        else:
            # Self-contained anomaly detection
            for idx, row in test_matrix.iterrows():
                row_mean = row.mean()
                row_std = row.std() + 1e-6
                z_scores = np.abs((row - row_mean) / row_std)
                # Apply minimum_scale filter
                anomalous_values = row[(z_scores > self.z_threshold) & (row > self.config.minimum_scale)]
                if len(anomalous_values) > 0:
                    anomalies.add(idx)

        return anomalies

    def _detect_col_anomalies(self,
                              test_matrix: pd.DataFrame,
                              baseline_matrix: Optional[pd.DataFrame]) -> Set[int]:
        """Detect anomalies by analyzing columns (patterns)."""
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
            # Apply minimum_scale filter: only consider values above threshold
            anomaly_mask = (z_scores > self.z_threshold) & (col_data > self.config.minimum_scale)
            anomaly_rows = col_data.index[anomaly_mask].tolist()
            anomalies.update(anomaly_rows)

        return anomalies


class DeCoMa:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.extractor = PatternExtractor(config)
        self.detector = AnomalyDetector(config)
        self.baseline_matrix = None

    def train_baseline(self, clean_samples: list[CodeSample]) -> None:
        print(f"\n{'='*60}")
        print("PHASE 1: Establishing Baseline")
        print(f"{'='*60}")

        # Extract patterns
        pattern_sets = []
        for sample in tqdm(clean_samples, desc="Extracting patterns"):
            if sample.is_valid():
                pattern_set = self.extractor.extract(sample)
                pattern_sets.append(pattern_set)

        # Build baseline matrix
        matrix_builder = PatternMatrix(pattern_sets)
        self.baseline_matrix = matrix_builder.get_matrix()

        print(f"Baseline established with {len(clean_samples)} samples")
        print(f"Matrix shape: {self.baseline_matrix.shape}")

    def detect_poisoned(self, test_samples: list[CodeSample]) -> Tuple[Set[int], dict]:
        print(f"\n{'='*60}")
        print("PHASE 2: Detecting Poisoned Samples")
        print(f"{'='*60}")

        # Extract patterns
        pattern_sets = []
        valid_indices = []

        for idx, sample in enumerate(tqdm(test_samples, desc="Extracting patterns")):
            if sample.is_valid():
                pattern_set = self.extractor.extract(sample)
                pattern_sets.append(pattern_set)
                valid_indices.append(idx)
        
        # Build test matrix
        matrix_bulder = PatternMatrix(pattern_sets)
        test_matrix = matrix_bulder.get_matrix()

        print(f"Test matrix shape: {test_matrix.shape}")

        # Detect anomalies
        start_time = time.time()
        anomaly_indices = self.detector.detect(test_matrix, self.baseline_matrix)
        detection_time = time.time() - start_time

        # Map back to original indices
        poisoned_indices = {valid_indices[i] for i in anomaly_indices}

        # Calculate metrics
        metrics = {
            "total_samples": len(test_samples),
            "valid_samples": len(valid_indices),
            "detected_poisoned": len(poisoned_indices),
            "detection_rate": len(poisoned_indices) / len(test_samples) if test_samples else 0,
            "detection_time": detection_time
        }

        print(f"\nResults:")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Detected poisoned: {metrics['detected_poisoned']}")
        print(f"  Detection rate: {metrics['detection_rate']:.2%}")
        print(f"  Time: {metrics['detection_time']:.2f}s")

        return poisoned_indices, metrics

    @staticmethod
    def evaluate(predictions: list[int], labels: list[int]) -> dict[str, float]:
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
