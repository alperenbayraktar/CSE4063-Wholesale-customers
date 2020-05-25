from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import time

class FrequentPattern:
  def apriori(self, df, min_support, min_length):
    start_time = time.time()
    frequent_itemsets_apriori = apriori(df, min_support=min_support, use_colnames=True)
    end_time = time.time()
    print(end_time-start_time)
    frequent_itemsets_apriori['length'] = frequent_itemsets_apriori['itemsets'].apply(lambda x: len(x))
    frequent_itemsets_apriori = frequent_itemsets_apriori[~(frequent_itemsets_apriori['length'] <= min_length)]
    frequent_itemsets_apriori.to_csv('apriori_result.csv')
    frequent_itemsets_apriori['dataset'] = 'apriori'
    return frequent_itemsets_apriori

  def fpgrowth(self, df, min_support, min_length):
    start_time = time.time()
    frequent_itemsets_fpgrowth = fpgrowth(df, min_support=min_support, use_colnames=True)
    end_time = time.time()
    print(end_time-start_time)
    frequent_itemsets_fpgrowth['length'] = frequent_itemsets_fpgrowth['itemsets'].apply(lambda x: len(x))
    frequent_itemsets_fpgrowth = frequent_itemsets_fpgrowth[~(frequent_itemsets_fpgrowth['length'] <= min_length)]
    frequent_itemsets_fpgrowth.to_csv('fpgrowth_result.csv')
    frequent_itemsets_fpgrowth['dataset'] = 'fpgrowth'
    return frequent_itemsets_fpgrowth

  def eclat(self, frequent_itemsets, min_support):
    start_time = time.time()
    eclat_rules = association_rules(frequent_itemsets, metric='support', min_threshold=min_support, support_only=True)
    end_time = time.time()
    print(end_time-start_time)
    eclat_rules = eclat_rules.dropna(axis=1) # drop unused columns in association rules DataFrame
    eclat_rules['itemsets'] = [x | y for x, y in zip(eclat_rules['antecedents'], eclat_rules['consequents'])]
    eclat_rules["length"] = eclat_rules['itemsets'].apply(lambda x: len(x))
    eclat_rules = eclat_rules.drop(['antecedents', 'consequents'], axis=1)
    eclat_rules.to_csv('eclat_result.csv')
    eclat_rules['dataset'] = 'eclat'
    return eclat_rules
