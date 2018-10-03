import pandas as pd
from engarde import decorators as ed
import pathlib
import logging
import datetime
from concurrent import futures
import toolz
import tqdm
import numpy as np
import time


THIS_DIR = pathlib.Path(__file__).parent
logger = logging.getLogger(__name__)


def extract_table(date: datetime.date, url=None) -> pd.DataFrame:
    if url is None:
        url = (r"https://www.powersmartpricing.org/psp/servlet?"
               r"type=pricingtabledatesingle&date={}")
    time.sleep(np.abs(np.random.random()))
    table_list = pd.read_html(url.format(date.strftime(r"%Y%m%d")))
    df = table_list[0]

    df["date"] = date

    return df


def scrape_tables(
        start: datetime.date=datetime.date(2014, 5, 1),
        end: datetime.date=None) -> pd.DataFrame:
    """

    Parameters
    ----------
    start:
        Datetime start scraping from. No data is available prior to
        May 1, 2014
    end:
        Datetime to stop scraping. The next day's data is available
        after 430PM. There's logic to select the current day, or the
        next day depending on when this is requested.

    Returns
    -------
        pd.DataFrame:
            A pandas DataFrame of cleaned data.

    """
    if end is None:
        now = datetime.datetime.now()
        end = now.date() if now.time() <= datetime.time(16, 30) else now.date() + datetime.timedelta(days=1)

    dates = pd.date_range(start=start, end=end, freq="D")

    # raw_data = (extract_table(date) for date in dates)
    # clean_data = pd.concat(raw_data)

    with futures.ThreadPoolExecutor(4) as executor:
        raw_data = tqdm.tqdm(executor.map(extract_table, dates), total=len(dates))
        data_pipe_line = toolz.pipe(
            raw_data,
        )

        clean_data = pd.concat(data_pipe_line)

    return clean_data


def create_raw_hourly_energy_rate_data(
        start: datetime.date=datetime.date(2014, 5, 1),
        end: datetime.date=datetime.date(2018, 10, 2)) -> pd.DataFrame:
    if end is None:
        now = datetime.datetime.now()
        end = now.date() if now.time() <= datetime.time(16, 30) else now.date() + datetime.timedelta(days=1)

    dates = pd.date_range(start=start, end=end, freq="D")

    # raw_data = (extract_table(date) for date in dates)
    # clean_data = pd.concat(raw_data)

    with futures.ThreadPoolExecutor(4) as executor:
        raw_data = tqdm.tqdm(executor.map(extract_table, dates), total=len(dates))
        data_pipe_line = toolz.pipe(
            raw_data,
        )

        raw_data = pd.concat(data_pipe_line)

    return raw_data

@toolz.curry
def extract_price_per_kwh(df: pd.DataFrame, col_name) -> pd.DataFrame:
    """
    Raw price comes in as Cents per kWh and it's a string.
    Convert to $/kWh and cast to a float
    """
    pat = r"(\d+\.\d+)"
    df[col_name] = (
                       df["Actual Price (Cents per kWh)"]
                           .str
                           .extract(pat, expand=False)
                           .astype(float)
                   ) / 100
    return df


@toolz.curry
def add_price_start_time(df: pd.DataFrame, col_name) -> pd.DataFrame:
    start_dates = df["date"]
    start_times = (
        df["Time of Day (CT)"]
            .str
            .extract(r"(\d{1,2}:\d{2} [A|P]M) - \d{1,2}:\d{2} [A|P]M", expand=False)
    )

    start_dt = start_dates.combine(
        other=start_times,
        func=lambda date, time: datetime.datetime.combine(date, pd.Timestamp(time).time())
    )

    df[col_name] = start_dt

    return df


@toolz.curry
def add_price_end_time(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    stop_dates = df["date"]
    stop_times = (
        df["Time of Day (CT)"]
            .str
            .extract(r"\d{1,2}:\d{2} [A|P]M - (\d{1,2}:\d{2} [A|P]M)", expand=False)
    )

    stop_dt = stop_dates.combine(
        other=stop_times,
        func=lambda date, time: datetime.datetime.combine(date, pd.Timestamp(time).time())
    )

    df[col_name] = stop_dt - pd.Timedelta(seconds=1)

    return df


def create_clean_hourly_energy_rate_data() -> pd.DataFrame:
    raw_fid = THIS_DIR.joinpath("..", "..", "data", "raw", "hourly_power_rates.csv")
    try:
        raw_data = pd.read_csv(raw_fid, parse_dates=["date"])
    except FileNotFoundError:
        logger.info("Interim hourly rate data set not found, recreating.")
        raw_data = create_raw_hourly_energy_rate_data()
        raw_data.to_csv(raw_fid, index=False)

    cleaning_steps = [
        extract_price_per_kwh(col_name="price_dollar_per_kwh"),
        add_price_start_time(col_name="price_start_datetime"),
        add_price_end_time(col_name="price_end_datetime"),
        lambda x: pd.DataFrame.drop(x, labels=["Time of Day (CT)", "Actual Price (Cents per kWh)", "date"], axis=1)
    ]

    cleaned_data = toolz.pipe(raw_data, *cleaning_steps)

    return cleaned_data


@ed.none_missing(columns=["price_start_datetime", "price_end_datetime", "price_dollar_per_kwh"])
@ed.within_range({"price_dollar_per_kwh": (0, 0.25)})
def load_hourly_energy_rate_data(fid=None):
    if fid is None:
        fid = THIS_DIR.joinpath("..", "..", "data", "interim", "hourly_power_rates.csv")
    try:
        clean_data = pd.read_csv(fid, parse_dates=["price_start_datetime", "price_end_datetime"])
    except FileNotFoundError:
        logger.info("Processed hourly rate data set not found, recreating.")
        clean_data = create_clean_hourly_energy_rate_data()
        clean_data.to_csv(fid, index=False)

    return clean_data


def main():
    load_hourly_energy_rate_data(THIS_DIR.joinpath("..", "..", "data", "interim", "hourly_power_rates.csv"))


if __name__ == '__main__':
    main()
