import requests
import pandas as pd
import datetime as dt
from utils import upload_json, get_current_timestamp, DATABASE_NAME


def get_data():
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    }
    res = requests.get(
        "https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=headers
    ).json()
    date = res["data"]["date"]
    main_data = res["data"]["data"]["rows"]

    df = pd.DataFrame().from_dict(main_data)
    df = df[["symbol", "companyName", "marketCap", "lastSalePrice", "percentageChange"]]

    # fix company name
    df["companyName"] = df["companyName"].str.lower()

    black_list = [
        "common stock",
        "incorporated",
        "inc.",
        ", inc.",
        "ordinary shares",
        "plc",
        "corporation",
        "class a",
        "series a",
        "american depositary shares",
        "new york registry shares",
        "\\(de\\)",
        "n.v.",
    ]
    pattern = "|".join(black_list)
    df["companyName"] = df["companyName"].str.replace(pattern, "", regex=True)
    df["companyName"] = df["companyName"].str.replace("[ \t]{2,}", " ", regex=True)
    df["companyName"] = df["companyName"].str.strip()

    # convert to float
    df["marketCap"] = df["marketCap"].str.replace(",", "").astype(float)
    df["lastSalePrice"] = (
        df["lastSalePrice"].str.replace("(\\$)|(,)", "", regex=True).astype(float)
    )
    df["percentageChange"] = (
        df["percentageChange"]
        .str.replace("(\\%)", "", regex=True)
        .replace("UNCH", "0")
        .astype(float)
    )
    df["date"] = date

    try:
        df["date_f"] = dt.datetime.strptime(date, "%b %d, %Y").strftime("%Y-%m-%d")
        df["market_open"] = False
    except:
        df["date_f"] = dt.datetime.strptime(date, "%b %d, %Y %H:%M %p").strftime(
            "%Y-%m-%d %H:%M %p"
        )
        df["market_open"] = True
    df["execution_ts"] = get_current_timestamp()

    df = df.sort_values(by="marketCap", ascending=False)
    data = df.to_dict("records")
    upload_json(data, path=f"{DATABASE_NAME}/1-raw/nasdaq.json", extend=True)
    return data


if __name__ == "__main__":
    get_data()
