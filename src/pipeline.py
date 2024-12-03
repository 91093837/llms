import os
import numpy as np
import pandas as pd
from utils import load_json, upload_json
from nasdaq import get_data
from models.portfolio import run as run_portfolio
from models.ranking import run as run_ranking


def calculate_summary():
    pct_chg = load_json("database/1-raw/nasdaq.json")
    portfolio = load_json("database/3-reporting/portfolio.json")

    pct_chg = pd.DataFrame().from_dict(pct_chg)
    pct_chg = pct_chg.drop_duplicates(subset=["symbol", "date_f"])
    pct_chg = pct_chg.pivot(index="date_f", columns="symbol", values="percentageChange")

    def metrics(pnl):
        output = pd.Series()
        shifts = [1, 5, 21, 63, 126]
        for s in shifts:
            output[f"{s}d_sharpe"] = (
                pnl.rolling(s).mean().iloc[-1] / pnl.rolling(s).std().iloc[-1]
            )
            output[f"{s}d_vol"] = pnl.rolling(s).std().iloc[-1]
        output["max_drawdown"] = (pnl.cummax() - pnl).max()
        return output

    names = set(p["name"] for p in portfolio)

    output = []
    for n in names:
        w = {p["date"]: p["portfolio"] for p in portfolio if p["name"] == n}
        w = pd.DataFrame().from_dict(w).T
        pnl = (w * pct_chg).sum(axis=1)

        summary = metrics(pnl)
        summary["name"] = n

        new_index = ["name"] + summary.index.difference(["name"]).to_list()
        summary = summary.reindex(new_index)
        summary = summary.replace(np.nan, "NaN")

        output.append(summary.to_dict())
    upload_json(output, "database/3-reporting/ranking.json")
    return None


def parse_data():
    portfolio = load_json("database/3-reporting/portfolio.json")
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
    upload_json(last_portfolio, "database/3-reporting/last_portfolio.json")
    return None


def main():
    assert "llms" in os.getcwd()

    # get nasdaq market-data
    data = get_data()

    # call llms with today-tickers
    run_ranking(data)
    run_portfolio(data)

    # build portfolios & ranking
    parse_data()
    calculate_summary()
    return None


if __name__ == "__main__":
    main()
