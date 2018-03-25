import pandas as pd

def display_all(df):
    """
    Display all the rows and columns of the dataframe
    :param df: Dataframe
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)

def display_some(df,rows,cols):
    """
    Display a given number of rows and columns of the dataframe
    :param df: Dataframe
    :param rows: Number of rows
    :param cols: Number of columns
    """
    with pd.option_context("display.max_rows", rows, "display.max_columns", cols):
        display(df)