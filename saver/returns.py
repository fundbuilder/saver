import polars as pl
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from dataclasses import dataclass

TRADING_DAYS_PER_MONTH = 21
DEFAULT_RISK_FREE_RATE = 0.03


@dataclass
class MarketAllocation:
    percentile: float
    market_weight: float
    risk_free_weight: float
    expected_return_at_percentile: float


def compute_optimal_allocation(
    returns: pl.DataFrame,
    months: int,
    tolerance_pct: float,
    var_percentile: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> MarketAllocation:
    """
    Computes the allocation between Market and Risk-Free assets.

    Args:
        returns: Polars DataFrame containing historical rolling returns (decimal).
        tolerance_pct: The max loss allowed (e.g., -0.10 for -10%).
        risk_free_rate: Annualized risk-free rate (e.g., 0.03).
    """
    # We assume the first column contains the rolling period returns
    m_returns = returns[returns.columns[0]].to_numpy()

    market_return_at_percentile = np.percentile(m_returns, var_percentile)

    risk_free_return_over_period = (1 + risk_free_rate) ** (months / 12) - 1

    denominator = market_return_at_percentile - risk_free_return_over_period

    if denominator == 0:
        w = 0.0
    else:
        w = (tolerance_pct - risk_free_return_over_period) / denominator

    # No leverage (w <= 1.0) and no shorting (w >= 0.0).
    market_weight = max(0.0, min(1.0, w))
    risk_free_weight = 1.0 - market_weight

    return MarketAllocation(
        percentile=var_percentile,
        market_weight=market_weight,
        risk_free_weight=risk_free_weight,
        expected_return_at_percentile=(market_weight * market_return_at_percentile)
        + (risk_free_weight * risk_free_return_over_period),
    )


def get_returns_distribution_plot(
    df: pl.DataFrame, k_months: int, allocation: MarketAllocation, var_percentile: float
) -> go.Figure:
    k_trading_days = k_months * TRADING_DAYS_PER_MONTH
    returns_pct = [r * 100 for r in df[df.columns[0]].to_list()]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns_pct,
            nbinsx=200,
            name="Distribution",
            marker_color="#1f77b4",
            opacity=0.7,
            histnorm="probability density",
            hoverinfo="none",
        )
    )

    kde = stats.gaussian_kde(returns_pct)
    x_range = np.linspace(min(returns_pct), max(returns_pct), 200)
    kde_values = kde(x_range)

    cdf_values = np.array([kde.integrate_box_1d(-np.inf, x) for x in x_range])

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=kde_values,
            mode="lines",
            name="Density Curve",
            line=dict(color="#ff7f0e", width=3),
            customdata=np.column_stack((cdf_values * 100,)),
            hovertemplate=(
                "Return: %{x:.2f}%<br>CDF: %{customdata[0]:.2f}%<br><extra></extra>"
            ),
        )
    )

    danger_x = x_range[cdf_values <= var_percentile / 100]
    danger_y = kde_values[: len(danger_x)]

    fig.add_vline(
        x=max(danger_x),
        line_dash="dash",
        line_color="#d62728",
        line_width=2,
        annotation_text=f"{100 - var_percentile}% Confidence Level",
        annotation_position="top left",
    )

    fig.add_trace(
        go.Scatter(
            x=list(danger_x) + [danger_x[-1], danger_x[0]],
            y=list(danger_y) + [0, 0],
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"Worst {var_percentile}% Outcomes",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=f"Distribution of {k_months}-Month ({k_trading_days}-Day) Rolling Returns",
        xaxis_title="Return (%)",
        yaxis_title="Density",
        showlegend=True,
        height=500,
        hovermode="x",
    )

    return fig
