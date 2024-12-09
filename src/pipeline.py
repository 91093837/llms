import os
import numpy as np
import pandas as pd
import logging
from utils import load_json, upload_json, DATABASE_NAME, JSONFile
from nasdaq import get_data
from models.portfolio import run as run_portfolio
from models.ranking import run as run_ranking


from pathlib import Path

paths = [
    f"{DATABASE_NAME}/1-raw",
    f"{DATABASE_NAME}/2-model_output",
    f"{DATABASE_NAME}/3-reporting",
]
for path in paths:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_market_index():
    market_data = load_json(f"{DATABASE_NAME}/1-raw/nasdaq.json")
    portfolio = load_json(f"{DATABASE_NAME}/3-reporting/portfolio.json")

    market_data = pd.DataFrame().from_dict(market_data)

    market_data["nasdaq_index"] = (
        market_data.groupby("date_f")["marketCap"]
        .apply(lambda v: v.div(v.sum()))
        .values
    )
    market_data["equal_weight"] = (
        market_data.groupby("date_f")["marketCap"]
        .apply(lambda v: v.div(v).div(v.shape[0]))
        .values
    )

    output = []
    dates = market_data["date_f"].unique()
    for date in dates:
        date_slice = market_data.loc[market_data["date_f"] == date, :]
        for col in ["nasdaq_index", "equal_weight"]:
            output.append(
                {
                    "date": date,
                    "name": col,
                    "execution_ts": date_slice["execution_ts"].iloc[0],
                    "portfolio": date_slice.set_index("symbol")[col].to_dict(),
                }
            )

    portfolio += output
    JSONFile(portfolio, f"{DATABASE_NAME}/3-reporting/portfolio.json", extend=True)
    return None


def calculate_summary():
    pct_chg = load_json(f"{DATABASE_NAME}/1-raw/nasdaq.json")
    portfolio = load_json(f"{DATABASE_NAME}/3-reporting/portfolio.json")

    pct_chg = pd.DataFrame().from_dict(pct_chg)
    pct_chg = pct_chg.drop_duplicates(subset=["symbol", "date_f"])
    pct_chg = pct_chg.pivot(index="date_f", columns="symbol", values="percentageChange")

    def metrics(pnl):
        output = pd.Series()
        shifts = [1, 5, 21]
        output["n_days"] = pnl.shape[0]
        output["acm"] = (1 + pnl).cumprod().iloc[-1]
        for s in shifts:
            output[f"{s}d_sharpe"] = (
                pnl.rolling(s).mean().iloc[-1] / pnl.rolling(s).std().iloc[-1]
            )
            output[f"{s}d_vol"] = pnl.rolling(s).std().iloc[-1]
        output["max_drawdown"] = (pnl.cummax() - pnl).max()
        output = output.round(4)
        return output

    names = set(p["name"] for p in portfolio)

    output = []
    for n in names:
        weight = {p["date"]: p["portfolio"] for p in portfolio if p["name"] == n}
        weight = pd.DataFrame().from_dict(weight).T
        pnl = (weight * pct_chg / 100).sum(axis=1)

        summary = metrics(pnl)
        summary["name"] = n

        new_index = ["name"] + summary.index.difference(["name"], sort=False).to_list()
        summary = summary.reindex(new_index)
        summary = summary.replace(np.nan, "NaN").replace(0, "NaN")
        output.append(summary.to_dict())
    JSONFile(output, f"{DATABASE_NAME}/3-reporting/ranking.json")
    return None


def parse_data():
    portfolio = load_json(f"{DATABASE_NAME}/3-reporting/portfolio.json")
    last_date = max([p["date"] for p in portfolio])
    exclude_list = ["date", "name", "execution_ts"]
    last_portfolio = {
        p["name"]: [
            {"symbol": k, "weight": v}
            for k, v in p["portfolio"].items()
            if k not in exclude_list
        ]
        for p in portfolio
        if p["date"] == last_date
    }
    JSONFile(last_portfolio, f"{DATABASE_NAME}/3-reporting/last_portfolio.json")
    return None


def main():
    assert "llms" in os.getcwd()

    # # get nasdaq market-data
    data = get_data()

    if data[-1]["market_open"]:
        # # call llms with today-tickers
        run_ranking(data)
        run_portfolio(data)

        # build portfolios & ranking
        build_market_index()
        parse_data()
        calculate_summary()

    return None


if __name__ == "__main__":
    main()
