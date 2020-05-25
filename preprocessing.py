import pandas as pd

class PreProcessing:
    def get_stats(self, df):
        for column in df.columns:
            print('Statistics for Column' + str(column))
            print(str(df[column].describe()) + '\n')

    def convert_categorical(self,df, column, bins, labels):
        df[column] = pd.cut(df[column], bins=bins, labels=labels)
        return df
       
    def categorize(self, df):
        self.convert_categorical(df,"Channel",[0.5,1.5,2.5],['Hotel/Restaurant/Cafe','Retail'])
        self.convert_categorical(df,"Region",[0.5,1.5,2.5,3.5],['Lisbon','Oporto','Other'])
        self.convert_categorical(df,"Fresh",[0,12000,24000,36000,120000],['fre_very_low', 'fre_low','fre_medium','fre_high'])
        self.convert_categorical(df,"Milk",[0,2000,4000,8000,100000],['m_very_low', 'm_low','m_medium','m_high'])
        self.convert_categorical(df,"Grocery",[0,2000,4000,8000,100000],['g_very_low', 'g_low','g_medium','g_high'])
        self.convert_categorical(df,"Frozen",[0,2000,4000,8000,100000],['fro_very_low', 'fro_low','fro_medium','fro_high'])
        self.convert_categorical(df,"Detergents_Paper",[0,1000,2500,5000,50000],['dp_very_low', 'dp_low','dp_medium','dp_high'])
        self.convert_categorical(df,"Delicassen",[0,1000,2500,5000,50000],['d_very_low', 'd_low','d_medium','d_high'])