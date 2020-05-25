import pandas as pd
import seaborn as sns
import collections
from matplotlib import pyplot as plt

class Graphs:
    def convert_to_ordered_count_dict(self,data,dataset=None):
        if(dataset == None):
            count_df = data['length'].value_counts().to_dict()
            ordered_dict = collections.OrderedDict(sorted(count_df.items()))
        else:
            count_df = data.loc[data['dataset'] == dataset]['length'].value_counts().to_dict()
            ordered_dict = collections.OrderedDict(sorted(count_df.items()))
        print(ordered_dict)
        return ordered_dict

    def get_single_graph(self,dataset, filter_support= None, multiple_datasets=None):
        if(filter_support == None):
            filtered_data = dataset        
        else:
            filtered_data = dataset[dataset.support > filter_support]
        
        if(multiple_datasets == None):
            sns.catplot(x="length", y="support", data=filtered_data,s=4)
            ordered_dict = self.convert_to_ordered_count_dict(filtered_data)
        else:
            sns.catplot(x="length", y="support", hue= "dataset", data=filtered_data, legend=True, legend_out=True, s= 4)
            ordered_apiori_dict = self.convert_to_ordered_count_dict(filtered_data,"apriori")
            ordered_fpgrowth_dict = self.convert_to_ordered_count_dict(filtered_data,"fpgrowth")
            ordered_eclat_dict = self.convert_to_ordered_count_dict(filtered_data,"eclat")
        plt.show()

        if(multiple_datasets == None):
            ax = sns.countplot(y="length", data=filtered_data)
        else:
            ax = sns.countplot(y="length", data=filtered_data, hue="dataset")

        plt.title('Distribution of Lengths, for support > ' + str(filter_support))
        plt.xlabel('Count')

        i = 0
        j = 0
        z = 0
        flag = 1

        if(multiple_datasets == None):
            keys = list(ordered_dict.values())
        else:
            keys_apriori = list(ordered_apiori_dict.values())
            keys_fpgrowth = list(ordered_fpgrowth_dict.values())
            keys_eclat = list(ordered_eclat_dict.values())

        for p in ax.patches:
            if(multiple_datasets == None):
                count = keys[i]
                i += 1
            else:
                if(flag == 1):
                    count = keys_apriori[i]
                    i += 1
                    if(i == len(keys_apriori)): flag = 2        
                elif(flag == 2):
                    count = keys_fpgrowth[j]
                    j += 1       
                    if(j == len(keys_fpgrowth)): flag = 3   
                else:
                    count = keys_eclat[z]
                    z += 1            

            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height()/2 + 0.03
            ax.annotate(count, (x, y))
        plt.show()

    def get_multiple_graph(self,dataset_1, dataset_2, dataset_3, filter_support= None):
        
        frames = [dataset_1, dataset_2, dataset_3]
        dataset = pd.concat(frames)
        print(dataset)
        
        self.get_single_graph(dataset,multiple_datasets=True,filter_support=filter_support)

    def get_visual_data(self, df):    
        sns.catplot(x="Channel",kind='count',data=df)
        plt.show()    
        sns.catplot(x="Region",kind='count',data=df)
        plt.show()    
        sns.catplot(x="Fresh",kind='count',data=df)
        plt.show()    
        sns.catplot(x="Milk",kind='count',data=df)
        plt.show()    
        sns.catplot(x="Grocery",kind='count',data=df)
        plt.show()    
        sns.catplot(x="Frozen",kind='count',data=df)
        plt.show()    
        sns.catplot(x="Detergents_Paper",kind='count',data=df)
        plt.show()    
        sns.catplot(x="Delicassen",kind='count',data=df)
        plt.show()
