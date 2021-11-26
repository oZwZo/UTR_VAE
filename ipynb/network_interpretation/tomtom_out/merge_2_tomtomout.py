import sys
import pandas as pd

def sort_q_value(df):
    df.sort_values(['Query_ID','q-value'], ascending=True)
    df = df.drop_duplicates(['Query_ID'],keep='first')
    return df

df_name = sys.argv[1]
cv = sys.argv[2]

# read table
df1_p = f"{df_name}_ribo_cv{cv}.tsv"
df2_p = f"{df_name}_yeast_cv{cv}.tsv"
df1 = pd.read_table(df1_p, sep='\t').iloc[:-3]
df2 = pd.read_table(df2_p, sep='\t').iloc[:-3]

hr=sort_q_value(df1)
hy=sort_q_value(df2)

mdf =  hr[['Query_ID','q-value']].merge(hy[['Query_ID','q-value']], left_on=['Query_ID'], right_on=['Query_ID'], suffixes=['_ribo','_yeast'])

mdf.to_csv(f"{df_name}_overlapQ_cv{cv}.csv", index=False)