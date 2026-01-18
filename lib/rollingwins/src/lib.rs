use pyo3::prelude::*;
use pyo3_polars::export::polars_core::frame::DataFrame;
use pyo3_polars::export::polars_core::prelude::NamedFrom;
use pyo3_polars::export::polars_core::series::Series;
use pyo3_polars::PyDataFrame;

/// Calculate rolling K-day returns directly from a Polars DataFrame
#[pyfunction]
fn calculate_rolling_returns_df(
    pydf: PyDataFrame,
    column: &str,
    k: usize,
) -> PyResult<PyDataFrame> {
    let df = &pydf.0;

    let prices = df
        .column(column)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Column '{}' not found: {}",
                column, e
            ))
        })?
        .f64()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Column must be numeric: {}",
                e
            ))
        })?;

    if prices.len() < k {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Not enough data points. Need at least {} prices, got {}",
            k,
            prices.len()
        )));
    }

    // Calculate rolling returns
    let mut returns = Vec::with_capacity(prices.len() - k);

    for i in 0..=(prices.len() - k) {
        let start_price = prices.get(i).unwrap();
        let end_price = prices.get(i + k - 1).unwrap();

        let return_pct = (end_price - start_price) / start_price;
        returns.push(return_pct);
    }

    let series_name = format!("{}_return_{}_day", column, k);
    let result = DataFrame::new(vec![Series::new(series_name.into(), &returns).into()])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(result))
}

#[pymodule]
fn rollingwins(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_rolling_returns_df, m)?)?;
    Ok(())
}
