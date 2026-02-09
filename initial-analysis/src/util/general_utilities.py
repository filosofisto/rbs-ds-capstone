import pandas as pd
import logging

def overview_dataset(df: pd.DataFrame, head_count=10):
    info(f"Dataset Overview (info, describe, head({head_count})")
    print(df.info())
    print(df.describe())
    print(df.head(head_count))

def title(p_title: str):
    info("-----------------------------------")
    info(p_title)
    info("-----------------------------------")


# ------------------------------------------------
# Logging functions
# ------------------------------------------------

logging.basicConfig(
    level=logging.INFO,              # or DEBUG, WARNING, ERROR, CRITICAL
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def debug(text: str):
    logging.debug(text)

def info(text: str):
    logging.info(text)

def warning(text: str):
    logging.warning(text)

def error(text: str):
    logging.error(text)

def critical(text: str):
    logging.critical(text)