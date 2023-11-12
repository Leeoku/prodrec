import pandas as pd


def import_data():
    df1 = pd.read_csv("data/styles.csv")
    df2 = pd.read_csv("data/images.csv")
    result = pd.concat([df1, df2], axis=1)
    print(result)
