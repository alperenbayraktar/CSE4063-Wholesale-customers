import pandas as pd

class PreProcessing:
    def get_stats(self, df):
        for column in df.columns:
            print('Statistics for Column' + str(column))
            print(str(df[column].describe()) + '\n')

    def convert_categorical(self,df, column, bins, labels):
        df[column] = pd.cut(df[column], bins=bins, labels=labels)
        return df