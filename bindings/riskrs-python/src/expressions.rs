#![allow(clippy::unused_unit)]

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use riskrs_core::calculate_auc;

fn auc_output_type(_: &[Field]) -> PolarsResult<Field> {
    let auc_field = Field::new("auc".into(), DataType::Float64);
    let fpr_field = Field::new("fpr".into(), DataType::List(Box::new(DataType::Float64)));
    let tpr_field = Field::new("tpr".into(), DataType::List(Box::new(DataType::Float64)));
    let thresholds_field = Field::new(
        "thresholds".into(),
        DataType::List(Box::new(DataType::Float64)),
    );

    let fields = vec![auc_field, fpr_field, tpr_field, thresholds_field];
    Ok(Field::new("".into(), DataType::Struct(fields)))
}

#[polars_expr(output_type_func=auc_output_type)]
fn auc(inputs: &[Series]) -> PolarsResult<Series> {
    let y_true = inputs[0].bool()?;
    let y_score = inputs[1].f64()?;

    // Use the core library function
    match calculate_auc(y_true, y_score, Some(100)) {
        Some((auc_value, fpr, tpr, thresholds)) => {
            // Convert vectors to Polars Series
            let auc_series = Series::new("auc".into(), vec![auc_value]);
            let fpr_series = Series::new("fpr".into(), vec![Series::new("".into(), fpr)]);
            let tpr_series = Series::new("tpr".into(), vec![Series::new("".into(), tpr)]);
            let thresholds_series = Series::new(
                "thresholds".into(),
                vec![Series::new("".into(), thresholds)],
            );

            // Create struct
            let struct_series = StructChunked::from_series(
                "".into(),
                1,
                [auc_series, fpr_series, tpr_series, thresholds_series].iter(),
            )?;

            Ok(struct_series.into_series())
        }
        None => {
            // Return null struct for invalid input
            let null_auc = Series::new("auc".into(), vec![Option::<f64>::None]);
            let null_fpr = Series::new("fpr".into(), vec![Option::<Series>::None]);
            let null_tpr = Series::new("tpr".into(), vec![Option::<Series>::None]);
            let null_thresholds = Series::new("thresholds".into(), vec![Option::<Series>::None]);

            let struct_series = StructChunked::from_series(
                "".into(),
                1,
                [null_auc, null_fpr, null_tpr, null_thresholds].iter(),
            )?;

            Ok(struct_series.into_series())
        }
    }
}
