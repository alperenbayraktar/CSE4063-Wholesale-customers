import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from preprocessing import PreProcessing

##Use your dataset to construct 3 frequent pattern mining models as follows:
#i) Apriori.
#ii) FP-Growth.
#iii) ECLAT.
##Use your dataset to construct 3 clustering analysis methods as follows:
#i) K-Means.
#ii) AGNES.
#iii) DBSCAN.

cols = ["Channel","Region","Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]
df = pd.read_csv("data.csv",usecols=cols)
#print(df.describe())
pp = PreProcessing()






def get_visual_data():
    pp.convert_categorical(df,"Channel",[0.5,1.5,2.5],['CH1','CH2'])
    sns.catplot(x="Channel",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Region",[0.5,1.5,2.5,3.5],['R1','R2','R3'])
    sns.catplot(x="Region",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Fresh",[0,12000,24000,36000,120000],['FRE1','FRE2','FRE3','FRE4'])
    sns.catplot(x="Fresh",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Milk",[0,2000,4000,8000,100000],['M1','M2','M3','M4'])
    sns.catplot(x="Milk",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Grocery",[0,2000,4000,8000,100000],['G1','G2','G3','G4'])
    sns.catplot(x="Grocery",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Frozen",[0,2000,4000,8000,100000],['FRO1','FRO2','FRO3','FRO4'])
    sns.catplot(x="Frozen",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Detergents_Paper",[0,1000,2500,5000,50000],['DP1','DP2','DP3','DP4'])
    sns.catplot(x="Detergents_Paper",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Delicassen",[0,1000,2500,5000,50000],['D1','D2','D3','D4'])
    sns.catplot(x="Delicassen",kind='count',data=df)
    #plt.show()

  
get_visual_data()

from mlxtend.preprocessing import TransactionEncoder
#from mlxtend.frequent_patterns import apriori
from apyori import apriori
#te = TransactionEncoder()

#te_ary = te.fit(df.to_numpy()).transform(df.to_numpy())
#df = pd.DataFrame(te_ary, columns=te.columns_)
#
##print(df)frequent_itemsets = apriori(df, min_support=0.3, use_colna4es=True)
#
#frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
records = []
for i in range(0, 439):
    records.append([str(df.values[i, j]) for j in range(0, 7)])
association_rules = apriori(records, min_support=0.1, min_confidence=0.2, min_lift=3, min_length=4)
association_results = list(association_rules)
print(association_results)