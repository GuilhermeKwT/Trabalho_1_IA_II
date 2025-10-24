import pandas as pd

df = pd.read_csv('wine_quality_merged_quality.csv')

def find_ranges(df):
    ranges = {}
    for column in df.columns:
        if column != 'quality':
            min_val = df[column].min()
            max_val = df[column].max()
            ranges[column] = (min_val, max_val)
    return ranges

gene_bounds = find_ranges(df)
for gene, (min_val, max_val) in gene_bounds.items():
    print(f"{gene}: ({min_val}, {max_val})")
    