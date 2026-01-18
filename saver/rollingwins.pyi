import polars as pl

def calculate_rolling_returns_df(
    df: pl.DataFrame, col_name: str, window: int
) -> pl.DataFrame:
    """Calculates rolling returns using the Rust backend."""
    ...
