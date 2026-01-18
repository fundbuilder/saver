import logging
from pathlib import Path
import polars as pl
from shiny import App, ui, reactive, render
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget
from saver.returns import (
    get_returns_distribution_plot,
    compute_optimal_allocation,
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_MONTH,
)
import saver.rollingwins


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
    ui.panel_title("Investment Risk Optimizer"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Your Constraints",
                    ui.input_numeric(
                        "invest_months", "Investment Horizon (months):", value=36, min=1
                    ),
                    ui.input_slider(
                        "max_loss",
                        "Max Acceptable Loss (%):",
                        min=0,
                        max=50,
                        value=10,
                        step=1,
                    ),
                    ui.input_select(
                        "var_percentile",
                        "Confidence Level:",
                        choices={
                            "1.0": "99% (Conservative)",
                            "5.0": "95% (Moderate)",
                            "10.0": "90% (Aggressive)",
                        },
                        selected="1.0",
                    ),
                ),
                ui.accordion_panel(
                    "Market & Assumptions",
                    ui.input_date_range(
                        "date_range",
                        "Historical Data Range:",
                        start=df["Date"].min(),
                        end=df["Date"].max(),
                    ),
                    ui.input_numeric(
                        "rf_rate",
                        "Annual Risk-Free Rate (%):",
                        value=DEFAULT_RISK_FREE_RATE * 100,
                        step=0.1,
                    ),
                ),
            ),
            ui.input_action_button(
                "calc_returns", "Calculate Optimal Mix", class_="btn-primary w-100"
            ),
        ),
        ui.card(
            ui.card_header("S&P 500 Historical Returns Distribution"),
            output_widget("returns_dist_plot"),
        ),
        ui.layout_columns(
            ui.value_box(
                "Recommended Allocation",
                ui.output_text("allocation_text"),
                showcase=ui.tags.i(class_="bi bi-pie-chart"),
                theme="primary",
            ),
        ),
    ),
)


def server(input, output, session) -> None:
    @reactive.calc
    def filtered_df() -> pl.DataFrame:
        return df.filter(
            (pl.col("Date") >= input.date_range()[0])
            & (pl.col("Date") <= input.date_range()[1])
        )

    @reactive.calc
    def rolling_returns_df() -> pl.DataFrame:
        k_trading_days = input.invest_months() * TRADING_DAYS_PER_MONTH
        return saver.rollingwins.calculate_rolling_returns_df(
            filtered_df(), "Close", k_trading_days
        )

    @reactive.calc
    @reactive.event(input.calc_returns)
    def optimal_allocation_data():
        tolerance = -(input.max_loss() / 100)
        percentile = float(input.var_percentile())
        return compute_optimal_allocation(
            rolling_returns_df(),
            input.invest_months(),
            tolerance,
            percentile,
            input.rf_rate() / 100,
        )

    @render_widget
    @reactive.event(input.calc_returns)
    def returns_dist_plot() -> go.Figure:
        try:
            return get_returns_distribution_plot(
                rolling_returns_df(),
                input.invest_months(),
                optimal_allocation_data(),
                float(input.var_percentile()),
            )

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
    def allocation_text():
        res = optimal_allocation_data()
        return f"{res.market_weight * 100:.1f}% S&P 500 / {res.risk_free_weight * 100:.1f}% Cash"


app = App(app_ui, server)
