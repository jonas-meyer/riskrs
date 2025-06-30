use core::f64;
use polars::prelude::*;

/// Prepare sorted pairs of (score, label) for ROC calculation
///
/// Sorts by score in descending order, breaking ties by putting positive labels first
pub fn prepare_sorted_pairs(y_true: &BooleanChunked, y_score: &Float64Chunked) -> Vec<(f64, bool)> {
    let mut pairs = y_score
        .iter()
        .zip(y_true.iter())
        .filter_map(|(score_opt, label_opt)| match (score_opt, label_opt) {
            (Some(score), Some(label)) => Some((score, label)),
            _ => None,
        })
        .collect::<Vec<_>>();

    if pairs.is_empty() {
        return vec![];
    }

    // Sort by score descending, breaking ties by putting positive labels first
    pairs.sort_unstable_by(|a, b| match b.0.partial_cmp(&a.0) {
        Some(std::cmp::Ordering::Equal) => b.1.cmp(&a.1), // true > false for ties
        Some(ordering) => ordering,
        None => std::cmp::Ordering::Equal,
    });

    pairs
}

/// Downsample ROC curve points to a maximum number of points
///
/// Uses uniform sampling to reduce the number of points while preserving
/// the start and end points of the curve
pub fn downsample_roc_points(
    fpr: Vec<f64>,
    tpr: Vec<f64>,
    thresholds: Vec<f64>,
    max_points: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if fpr.len() <= max_points {
        return (fpr, tpr, thresholds);
    }

    let mut downsampled_fpr = Vec::with_capacity(max_points);
    let mut downsampled_tpr = Vec::with_capacity(max_points);
    let mut downsampled_thresholds = Vec::with_capacity(max_points);

    // Always include the first point (0, 0)
    downsampled_fpr.push(fpr[0]);
    downsampled_tpr.push(tpr[0]);
    downsampled_thresholds.push(thresholds[0]);

    // Sample intermediate points uniformly
    let step_size = (fpr.len() - 2) as f64 / (max_points - 2) as f64;

    for i in 1..max_points - 1 {
        let index = (i as f64 * step_size).round() as usize + 1;
        downsampled_fpr.push(fpr[index]);
        downsampled_tpr.push(tpr[index]);
        downsampled_thresholds.push(thresholds[index]);
    }

    // Always include the last point (1, 1)
    if let (Some(&last_fpr), Some(&last_tpr), Some(&last_threshold)) =
        (fpr.last(), tpr.last(), thresholds.last())
    {
        downsampled_fpr.push(last_fpr);
        downsampled_tpr.push(last_tpr);
        downsampled_thresholds.push(last_threshold);
    }

    (downsampled_fpr, downsampled_tpr, downsampled_thresholds)
}

/// Compute ROC curve points and AUC from sorted pairs
///
/// Returns (auc_value, fpr_vec, tpr_vec, thresholds_vec)
pub fn compute_roc_points(
    pairs: &[(f64, bool)],
    max_points: usize,
) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_pos = pairs.iter().filter(|(_, label)| *label).count();
    let n_neg = pairs.len() - n_pos;

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;

    // Group consecutive pairs with the same score
    let mut unique_scores: Vec<f64> = Vec::new();
    let mut tp_counts: Vec<usize> = Vec::new();
    let mut fp_counts: Vec<usize> = Vec::new();

    let mut current_score = f64::INFINITY;
    let mut current_tp = 0;
    let mut current_fp = 0;

    for (score, is_positive) in pairs {
        if *score != current_score {
            // Save the previous group if it exists
            if current_score != f64::INFINITY {
                unique_scores.push(current_score);
                tp_counts.push(current_tp);
                fp_counts.push(current_fp);
            }

            // Start new group
            current_score = *score;
            current_tp = 0;
            current_fp = 0;
        }

        if *is_positive {
            current_tp += 1;
        } else {
            current_fp += 1;
        }
    }

    // Don't forget the last group
    if current_score != f64::INFINITY {
        unique_scores.push(current_score);
        tp_counts.push(current_tp);
        fp_counts.push(current_fp);
    }

    // Build ROC curve points
    let mut fpr = Vec::with_capacity(unique_scores.len() + 2);
    let mut tpr = Vec::with_capacity(unique_scores.len() + 2);
    let mut thresholds = Vec::with_capacity(unique_scores.len() + 2);

    // Start at (0, 0) with infinite threshold
    fpr.push(0.0);
    tpr.push(0.0);
    thresholds.push(f64::INFINITY);

    let mut cumulative_tp = 0;
    let mut cumulative_fp = 0;

    for i in 0..unique_scores.len() {
        cumulative_tp += tp_counts[i];
        cumulative_fp += fp_counts[i];

        let current_tpr = cumulative_tp as f64 / n_pos_f;
        let current_fpr = cumulative_fp as f64 / n_neg_f;

        fpr.push(current_fpr);
        tpr.push(current_tpr);
        thresholds.push(unique_scores[i]);
    }

    // Calculate AUC using trapezoidal rule
    let mut auc_value = 0.0;
    for i in 1..fpr.len() {
        let width = fpr[i] - fpr[i - 1];
        let height = (tpr[i] + tpr[i - 1]) / 2.0;
        auc_value += width * height;
    }

    // Downsample if needed
    let (downsampled_fpr, downsampled_tpr, downsampled_thresholds) =
        downsample_roc_points(fpr, tpr, thresholds, max_points);

    (
        auc_value,
        downsampled_fpr,
        downsampled_tpr,
        downsampled_thresholds,
    )
}

/// Calculate AUC for binary classification
///
/// This is a convenience function that combines all the steps
pub fn calculate_auc(
    y_true: &BooleanChunked,
    y_score: &Float64Chunked,
    max_points: Option<usize>,
) -> Option<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let pairs = prepare_sorted_pairs(y_true, y_score);

    if pairs.is_empty() {
        return None;
    }

    let n_pos = pairs.iter().filter(|(_, label)| *label).count();
    let n_neg = pairs.len() - n_pos;

    if n_pos == 0 || n_neg == 0 {
        return None;
    }

    let max_pts = max_points.unwrap_or(100);
    let (auc_value, fpr, tpr, thresholds) = compute_roc_points(&pairs, max_pts);

    Some((auc_value, fpr, tpr, thresholds))
}

// ... existing code ...

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_classifier() {
        let pairs = vec![(0.9, true), (0.8, true), (0.3, false), (0.1, false)];
        let (auc, _fpr, _tpr, _thresholds) = compute_roc_points(&pairs, 100);
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_classifier() {
        let pairs = vec![(0.6, true), (0.4, false), (0.7, false), (0.3, true)];
        let (auc, _fpr, _tpr, _thresholds) = compute_roc_points(&pairs, 100);
        assert!((auc - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_downsampling() {
        let fpr = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let tpr = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let thresholds = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];

        let (down_fpr, down_tpr, down_thresh) = downsample_roc_points(fpr, tpr, thresholds, 5);

        assert_eq!(down_fpr.len(), 5);
        assert_eq!(down_tpr.len(), 5);
        assert_eq!(down_thresh.len(), 5);
        assert_eq!(down_fpr[0], 0.0);
        assert_eq!(down_fpr.last(), Some(&1.0));
    }

    #[test]
    fn test_prepare_sorted_pairs() {
        use polars::prelude::*;

        let y_true = BooleanChunked::from_slice("y_true".into(), &[true, false, true, false]);
        let y_score = Float64Chunked::from_slice("y_score".into(), &[0.9, 0.3, 0.8, 0.1]);

        let pairs = prepare_sorted_pairs(&y_true, &y_score);

        // Should be sorted by score descending
        assert_eq!(pairs[0], (0.9, true));
        assert_eq!(pairs[1], (0.8, true));
        assert_eq!(pairs[2], (0.3, false));
        assert_eq!(pairs[3], (0.1, false));
    }

    #[test]
    fn test_calculate_auc_integration() {
        use polars::prelude::*;

        let y_true = BooleanChunked::from_slice("y_true".into(), &[true, true, false, false]);
        let y_score = Float64Chunked::from_slice("y_score".into(), &[0.9, 0.8, 0.3, 0.1]);

        let result = calculate_auc(&y_true, &y_score, Some(100));
        assert!(result.is_some());

        let (auc, _fpr, _tpr, _thresholds) = result.unwrap();
        assert!((auc - 1.0).abs() < 1e-10); // Perfect classifier
    }
}
