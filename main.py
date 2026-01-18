import logging
from pathlib import Path
import polars as pl
from shiny import App, ui, render, reactive
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from shinywidgets import output_widget, render_widget

import rollingwins

TRADING_DAYS_PER_MONTH = 21


def load_sp500_data(filepath: str | Path) -> pl.DataFrame:
    """
    Load S&P 500 historical data and select Date and Close columns.

    Args:
        filepath: Path to the CSV file containing S&P 500 data

    Returns:
        DataFrame with Date and Close columns
    """
    df = pl.read_csv(
        filepath,
        skip_rows=3,
        has_header=False,
        new_columns=[
            "Date",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
        ],
    )

    return df.select([pl.col("Date").str.to_date(), pl.col("Close")])


data_file: Path = Path(__file__).parent / "data" / "sp500_historical.csv"

df: pl.DataFrame = load_sp500_data(data_file)

app_ui = ui.page_fluid(
    ui.panel_title("S&P 500 Historical Data"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_date_range(
                "date_range",
                "Date Range:",
                start=df["Date"].min(),
                end=df["Date"].max(),
            ),
            ui.input_slider(
                "k_days",
                "Rolling Window (# of months):",
                min=1,
                max=60,
                value=36,
                step=1,
            ),
            ui.input_checkbox("show_grid", "Show Grid", value=True),
            ui.input_action_button(
                "calc_returns", "Calculate Rolling Returns", class_="btn-primary"
            ),
        ),
        ui.output_ui("price_plot"),
        ui.output_text("summary_text"),
        output_widget("returns_dist_plot"),
    ),
)


def server(input, output, session) -> None:
    @render.ui
    def price_plot() -> ui.TagList:
        filtered_df = df.filter(
            (pl.col("Date") >= input.date_range()[0])
            & (pl.col("Date") <= input.date_range()[1])
        )

        data_rows = [
            f'[new Date("{row[0].isoformat()}"), {row[1]}]'
            for row in filtered_df.iter_rows()
        ]
        data_js = ",\n".join(data_rows)

        return ui.TagList(
            ui.head_content(
                ui.tags.link(
                    rel="stylesheet",
                    href="https://cdnjs.cloudflare.com/ajax/libs/dygraph/2.2.1/dygraph.min.css",
                ),
                ui.tags.script(
                    src="https://cdnjs.cloudflare.com/ajax/libs/dygraph/2.2.1/dygraph.min.js"
                ),
            ),
            ui.tags.div(id="graphdiv", style="width:100%; height:400px;"),
            ui.tags.script(f"""
                // We use a small delay to ensure the div is in the DOM
                setTimeout(function() {{
                    new Dygraph(
                        document.getElementById("graphdiv"),
                        [{data_js}],
                        {{
                            labels: ['Date', 'Close Price'],
                            showRangeSelector: true,
                            title: 'S&P 500'
                        }}
                    );
                }}, 50);
            """),
        )

    @render_widget
    @reactive.event(input.calc_returns)
    def returns_dist_plot() -> go.Figure:
        filtered_df = df.filter(
            (pl.col("Date") >= input.date_range()[0])
            & (pl.col("Date") <= input.date_range()[1])
        )

        # Convert months to trading days (approximately 21 trading days per month)
        k_months = input.k_days()
        k_trading_days = k_months * TRADING_DAYS_PER_MONTH
        try:
            returns: pl.DataFrame = rollingwins.calculate_rolling_returns_df(
                filtered_df, "Close", k_trading_days
            )
            returns_pct = [
                r * 100 for r in returns[returns.columns[0]].to_list()
            ]

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
                        "Return: %{x:.2f}%<br>"
                        "CDF: %{customdata[0]:.2f}%<br>"
                        "<extra></extra>"
                    ),
                )
            )

            mean_return = sum(returns_pct) / len(returns_pct)
            fig.add_vline(
                x=mean_return,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_return:.2f}%",
                annotation_position="top",
            )

            fig.update_layout(
                title=f"Distribution of {k_months}-Month ({k_trading_days}-Day) Rolling Returns",
                xaxis_title="Return (%)",
                yaxis_title="Density",
                showlegend=True,
                height=500,
                hovermode="x unified",
            )

            return fig

        except Exception as e:
            logging.error("error calculating cdf of returns %s", e)
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=500)
            return fig

    @render.text
    def summary_text() -> str:
        filtered_df = df.filter(
            (pl.col("Date") >= input.date_range()[0])
            & (pl.col("Date") <= input.date_range()[1])
        )

        min_price = filtered_df["Close"].min()
        max_price = filtered_df["Close"].max()
        avg_price = filtered_df["Close"].mean()

        return f"Min: {min_price:.2f} | Max: {max_price:.2f} | Average: {avg_price:.2f}"


app = App(app_ui, server)
