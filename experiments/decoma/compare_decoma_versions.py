"""
Compare the two DeCoMa implementations
"""

import sys
import numpy as np
from pathlib import Path

# Import both versions
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.caw.attackers.decoma.decoma import DeCoMa as DeCoMaSimple
from src.caw.attackers.decoma.decoma import DetectionConfig as ConfigSimple
from src.caw.attackers.decoma.utils import CodeSample

# Also import the complex version
from experiments.decoma.decoma_pair import DeCoMa as DeCoMaComplex
from experiments.decoma.decoma_pair import DetectionConfig as ConfigComplex
from experiments.decoma.decoma_pair import CodeSample as CodeSampleComplex


def create_test_data():
    """Create test data for comparison."""

    # Clean samples - normal patterns
    clean_samples_simple = []
    clean_samples_complex = []

    for i in range(20):
        # Simple version
        clean_samples_simple.append(CodeSample(
            code=f"def func{i}(x): return x + {i}",
            expressions=["def func ( x )", "return x +"],
            identifiers=["func", "x"]
        ))

        # Complex version (same data)
        clean_samples_complex.append(CodeSampleComplex(
            code=f"def func{i}(x): return x + {i}",
            expressions=["def func ( x )", "return x +"],
            identifiers=["func", "x"]
        ))

    # Test samples - mix of normal and anomalous
    test_samples_simple = []
    test_samples_complex = []

    # Normal samples
    for i in range(5):
        test_samples_simple.append(CodeSample(
            code=f"def helper{i}(y): return y * {i}",
            expressions=["def helper ( y )", "return y *"],
            identifiers=["helper", "y"]
        ))

        test_samples_complex.append(CodeSampleComplex(
            code=f"def helper{i}(y): return y * {i}",
            expressions=["def helper ( y )", "return y *"],
            identifiers=["helper", "y"]
        ))

    # Anomalous samples - different patterns
    anomalous_codes = [
        ("exec(payload)", ["exec ( payload )"], ["payload"]),
        ("eval(cmd)", ["eval ( cmd )"], ["cmd"]),
        ("os.system(shell)", ["os system ( shell )"], ["shell"]),
    ]

    for code, exprs, ids in anomalous_codes:
        test_samples_simple.append(CodeSample(
            code=code,
            expressions=exprs,
            identifiers=ids
        ))

        test_samples_complex.append(CodeSampleComplex(
            code=code,
            expressions=exprs,
            identifiers=ids
        ))

    return (clean_samples_simple, test_samples_simple,
            clean_samples_complex, test_samples_complex)


def test_simple_version(clean_samples, test_samples):
    """Test the simple version."""
    print("="*60)
    print("Testing SIMPLE Version (src/caw/attackers/decoma/)")
    print("="*60)

    config = ConfigSimple(
        z_score_threshold=2.0,
        detect_type="both"
    )

    detector = DeCoMaSimple(config)
    detector.train_baseline(clean_samples)
    poisoned_indices, metrics = detector.detect_poisoned(test_samples)

    return poisoned_indices, metrics


def test_complex_version(clean_samples, test_samples):
    """Test the complex version."""
    print("\n" + "="*60)
    print("Testing COMPLEX Version (experiments/)")
    print("="*60)

    config = ConfigComplex(
        z_score_threshold=2.0,
        detect_type="both",
        match_types=["pattern2pattern", "pattern2variable",
                    "variable2pattern", "variable2variable"]
    )

    detector = DeCoMaComplex(config)
    detector.train_baseline(clean_samples)
    poisoned_indices, metrics = detector.detect_poisoned(test_samples)

    # Also report clean_pairs
    print(f"\nClean pairs found: {len(detector.clean_pairs)}")

    return poisoned_indices, metrics


def main():
    """Compare both versions."""

    # Create test data
    (clean_simple, test_simple,
     clean_complex, test_complex) = create_test_data()

    print("Test Data Summary:")
    print(f"  Clean samples: {len(clean_simple)}")
    print(f"  Test samples: {len(test_simple)} (5 normal + 3 anomalous)")
    print(f"  Expected anomalous indices: {5, 6, 7}")

    # Test simple version
    simple_poisoned, simple_metrics = test_simple_version(
        clean_simple, test_simple
    )

    # Test complex version
    complex_poisoned, complex_metrics = test_complex_version(
        clean_complex, test_complex
    )

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print(f"\nSimple Version:")
    print(f"  Detected: {simple_poisoned}")
    print(f"  Detection rate: {simple_metrics['detection_rate']:.1%}")
    print(f"  Time: {simple_metrics['detection_time']:.4f}s")

    print(f"\nComplex Version:")
    print(f"  Detected: {complex_poisoned}")
    print(f"  Detection rate: {complex_metrics['detection_rate']:.1%}")
    print(f"  Time: {complex_metrics['detection_time']:.4f}s")

    print(f"\nExpected anomalous: {{5, 6, 7}}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    if not simple_poisoned and not complex_poisoned:
        print("❌ Both versions failed to detect anomalies")
        print("   Possible reasons:")
        print("   - Z-score threshold too high")
        print("   - Not enough variance in data")
        print("   - Anomalous patterns not distinctive enough")
    elif simple_poisoned and not complex_poisoned:
        print("✅ Simple version detected anomalies")
        print("❌ Complex version failed")
        print("   Simple version may be more robust for this type of data")
    elif complex_poisoned and not simple_poisoned:
        print("❌ Simple version failed")
        print("✅ Complex version detected anomalies")
        print("   Complex version may be better for subtle patterns")
    else:
        print("✅ Both versions detected anomalies")
        overlap = simple_poisoned & complex_poisoned
        print(f"   Overlap: {overlap}")
        print(f"   Simple only: {simple_poisoned - complex_poisoned}")
        print(f"   Complex only: {complex_poisoned - simple_poisoned}")


if __name__ == "__main__":
    main()