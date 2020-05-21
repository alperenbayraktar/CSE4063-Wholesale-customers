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

    pp.convert_categorical(df,"Fresh",[0,12000,24000,36000,120000],['fre_very_low', 'fre_low','fre_medium','fre_high'])
    sns.catplot(x="Fresh",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Milk",[0,2000,4000,8000,100000],['m_very_low', 'm_low','m_medium','m_high'])
    sns.catplot(x="Milk",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Grocery",[0,2000,4000,8000,100000],['g_very_low', 'g_low','g_medium','g_high'])
    sns.catplot(x="Grocery",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Frozen",[0,2000,4000,8000,100000],['fro_very_low', 'fro_low','fro_medium','fro_high'])
    sns.catplot(x="Frozen",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Detergents_Paper",[0,1000,2500,5000,50000],['dp_very_low', 'dp_low','dp_medium','dp_high'])
    sns.catplot(x="Detergents_Paper",kind='count',data=df)
    #plt.show()

    pp.convert_categorical(df,"Delicassen",[0,1000,2500,5000,50000],['d_very_low', 'd_low','d_medium','_high'])
    sns.catplot(x="Delicassen",kind='count',data=df)
    #plt.show()

  
get_visual_data()

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth
#from apyori import apriori
te = TransactionEncoder()
#df = df.drop(['Channel'], axis=1)
te_ary = te.fit(df.to_numpy()).transform(df.to_numpy())
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets_apriori = apriori(df, min_support=0.00001, use_colnames=True)

frequent_itemsets_apriori['length'] = frequent_itemsets_apriori['itemsets'].apply(lambda x: len(x))
frequent_itemsets_apriori = frequent_itemsets_apriori[~(frequent_itemsets_apriori['length'] <= 2)]
frequent_itemsets_apriori.to_csv('apriori_result.csv')
print(frequent_itemsets_apriori)
frequent_itemsets_fpgrowth = fpgrowth(df, min_support=0.00001, use_colnames=True)

frequent_itemsets_fpgrowth['length'] = frequent_itemsets_fpgrowth['itemsets'].apply(lambda x: len(x))
frequent_itemsets_fpgrowth = frequent_itemsets_fpgrowth[~(frequent_itemsets_fpgrowth['length'] <= 2)]
frequent_itemsets_fpgrowth.to_csv('apriori_result.csv')
print(frequent_itemsets_fpgrowth)
#records = []
#for i in range(0, 439):
#    records.append([str(df.values[i, j]) for j in range(0, 7)])
#association_rules = apriori(records, min_support=0.1, min_confidence=0.2, min_lift=3, min_length=1)
#association_results = list(association_rulesd)
#print(association_results)