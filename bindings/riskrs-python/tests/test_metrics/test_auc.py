import numpy as np
import polars as pl
import riskrs
from sklearn.metrics import roc_auc_score


def test_against_sklearn():
    """Compare our implementation with sklearn"""
    np.random.seed(42)
    y_true = np.random.choice([True, False], size=100)
    y_score = np.random.random(100)

    df = pl.DataFrame({"y_true": y_true, "y_score": y_score})
    result = df.select(riskrs.auc("y_true", "y_score").alias("result"))
    our_auc = result["result"].struct.field("auc")[0]

    sklearn_auc = roc_auc_score(y_true, y_score)

    tolerance = 1e-10
    diff = abs(our_auc - sklearn_auc)

    print(f"Our AUC: {our_auc:.10f}")
    print(f"Sklearn AUC: {sklearn_auc:.10f}")
    print(f"Difference: {diff:.10f}")

    assert diff < tolerance, (
        f"AUC mismatch too large: ours={our_auc}, sklearn={sklearn_auc}, diff={diff}"
    )


def test_perfect_classifier():
    """Test perfect classifier case"""
    df = pl.DataFrame(
        {"y_true": [True, True, False, False], "y_score": [0.9, 0.8, 0.3, 0.1]}
    )

    result = df.select(riskrs.auc("y_true", "y_score").alias("result"))
    auc_value = result["result"].struct.field("auc")[0]

    assert abs(auc_value - 1.0) < 1e-10, (
        f"Perfect classifier should have AUC = 1.0, got {auc_value}"
    )


def test_random_classifier():
    """Test that random classifier gives AUC around 0.5"""
    np.random.seed(123)
    y_true = np.random.choice([True, False], size=1000)
    y_score = np.random.random(1000)

    df = pl.DataFrame({"y_true": y_true, "y_score": y_score})
    result = df.select(riskrs.auc("y_true", "y_score").alias("result"))
    auc_value = result["result"].struct.field("auc")[0]

    # Random classifier should be around 0.5, give some tolerance
    assert 0.4 < auc_value < 0.6, (
        f"Random classifier AUC should be around 0.5, got {auc_value}"
    )


def test_roc_curve_properties():
    """Test ROC curve mathematical properties"""
    df = pl.DataFrame(
        {
            "y_true": [True, True, False, False, True],
            "y_score": [0.9, 0.8, 0.6, 0.4, 0.7],
        }
    )

    result = df.select(riskrs.auc("y_true", "y_score").alias("result"))

    fpr = result["result"].struct.field("fpr")[0]
    tpr = result["result"].struct.field("tpr")[0]

    print(f"FPR: {fpr}")
    print(f"TPR: {tpr}")

    fpr_vals = fpr.to_list()
    tpr_vals = tpr.to_list()

    assert all(fpr_vals[i] <= fpr_vals[i + 1] for i in range(len(fpr_vals) - 1)), (
        "FPR should be monotonic"
    )
    assert all(tpr_vals[i] <= tpr_vals[i + 1] for i in range(len(tpr_vals) - 1)), (
        "TPR should be monotonic"
    )

    assert fpr_vals[0] == 0.0 and tpr_vals[0] == 0.0, "Should start at (0,0)"
    assert fpr_vals[-1] == 1.0 and tpr_vals[-1] == 1.0, "Should end at (1,1)"


def test_invalid_input_returns_none():
    """Test that invalid inputs return None/null values"""
    # Test case 1: All labels are the same (no positive or no negative cases)
    df_all_positive = pl.DataFrame(
        {"y_true": [True, True, True, True], "y_score": [0.9, 0.8, 0.7, 0.6]}
    )

    result = df_all_positive.select(riskrs.auc("y_true", "y_score").alias("result"))
    auc_value = result["result"].struct.field("auc")[0]

    assert auc_value is None, f"All positive labels should return None, got {auc_value}"

    # Test case 2: All labels are negative
    df_all_negative = pl.DataFrame(
        {"y_true": [False, False, False, False], "y_score": [0.9, 0.8, 0.7, 0.6]}
    )

    result = df_all_negative.select(riskrs.auc("y_true", "y_score").alias("result"))
    auc_value = result["result"].struct.field("auc")[0]

    assert auc_value is None, f"All negative labels should return None, got {auc_value}"

    # Test case 3: Empty data (if this case occurs)
    df_empty = pl.DataFrame(
        {
            "y_true": pl.Series("y_true", [], dtype=pl.Boolean),
            "y_score": pl.Series("y_score", [], dtype=pl.Float64),
        }
    )

    result = df_empty.select(riskrs.auc("y_true", "y_score").alias("result"))
    auc_value = result["result"].struct.field("auc")[0]

    assert auc_value is None, f"Empty data should return None, got {auc_value}"


def test_full_struct_none_on_invalid_input():
    """Test that entire struct is null when calculation fails"""
    df_all_positive = pl.DataFrame(
        {"y_true": [True, True, True, True], "y_score": [0.9, 0.8, 0.7, 0.6]}
    )

    result = df_all_positive.select(riskrs.auc("y_true", "y_score").alias("result"))

    # Check all fields are null
    auc_value = result["result"].struct.field("auc")[0]
    fpr_value = result["result"].struct.field("fpr")[0]
    tpr_value = result["result"].struct.field("tpr")[0]
    thresholds_value = result["result"].struct.field("thresholds")[0]

    assert auc_value is None, "AUC should be None"
    assert fpr_value is None, "FPR should be None"
    assert tpr_value is None, "TPR should be None"
    assert thresholds_value is None, "Thresholds should be None"


def test_null_values_in_data():
    """Test handling of null values in input data"""
    df_with_nulls = pl.DataFrame(
        {
            "y_true": [True, False, None, True, False],
            "y_score": [0.9, 0.8, 0.7, None, 0.5],
        }
    )

    result = df_with_nulls.select(riskrs.auc("y_true", "y_score").alias("result"))
    auc_value = result["result"].struct.field("auc")[0]

    # Should still work with valid pairs, ignoring null values
    # In this case we have: (0.9, True), (0.8, False), (0.5, False)
    # This should give us a valid AUC since we have both positive and negative cases
    assert auc_value is not None, "Should handle null values gracefully"
    assert isinstance(auc_value, float), f"Should return float, got {type(auc_value)}"

    # Test all null case
    df_all_nulls = pl.DataFrame(
        {"y_true": [None, None, None], "y_score": [None, None, None]}
    )
    df_all_nulls = df_all_nulls.with_columns(
        pl.col("y_true").cast(pl.Boolean), pl.col("y_score").cast(pl.Float64)
    )

    result = df_all_nulls.select(riskrs.auc("y_true", "y_score").alias("result"))
    auc_value = result["result"].struct.field("auc")[0]

    assert auc_value is None, f"All null data should return None, got {auc_value}"
