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
    pp.convert_categorical(df,"Channel",[0.5,1.5,2.5],['1','2'])
    sns.catplot(x="Channel",kind='count',data=df)
    plt.show()

    pp.convert_categorical(df,"Region",[0.5,1.5,2.5,3.5],['1','2','3'])
    sns.catplot(x="Region",kind='count',data=df)
    plt.show()

    pp.convert_categorical(df,"Fresh",[0,12000,24000,36000,120000],['0-12000','12000-24000','24000-36000','36000-120000'])
    sns.catplot(x="Fresh",kind='count',data=df)
    plt.show()

    pp.convert_categorical(df,"Milk",[0,2000,4000,8000,100000],['0-2000','2000-4000','4000-8000','8000-100000'])
    sns.catplot(x="Milk",kind='count',data=df)
    plt.show()

    pp.convert_categorical(df,"Grocery",[0,2000,4000,8000,100000],['0-2000','2000-4000','4000-8000','8000-100000'])
    sns.catplot(x="Grocery",kind='count',data=df)
    plt.show()

    pp.convert_categorical(df,"Frozen",[0,2000,4000,8000,100000],['0-2000','2000-4000','4000-8000','8000-100000'])
    sns.catplot(x="Frozen",kind='count',data=df)
    plt.show()

    pp.convert_categorical(df,"Detergents_Paper",[0,1000,2500,5000,50000],['0-1000','1000-2500','2500-5000','5000-50000'])
    sns.catplot(x="Detergents_Paper",kind='count',data=df)
    plt.show()

    pp.convert_categorical(df,"Delicassen",[0,1000,2500,5000,50000],['0-1000','1000-2500','2500-5000','5000-50000'])
    sns.catplot(x="Delicassen",kind='count',data=df)
    plt.show()
    
#get_visual_data()