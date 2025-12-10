import pandas as pd
from sklearn.model_selection import train_test_split

# df contains numeric columns and 'country'
numeric_cols = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6']
country_col = 'country'

def make_strata_key(df, numeric_cols, n_bins=5):
    # will hold binned categorical columns
    binned_cols = []

    for c in numeric_cols:
        # qcut returns a categorical series aligned to df.index
        binned = pd.qcut(df[c], q=n_bins, duplicates='drop')
        binned_cols.append(binned.astype(str))  # safe conversion to string

    # concatenate into a DataFrame so alignment is preserved
    binned_df = pd.concat(binned_cols, axis=1)

    # join row values to form the composite key
    return binned_df.apply(lambda row: "|".join(row.values), axis=1)

# allocate empty holders
holdoutA_parts = []
holdoutB_parts = []
train_parts = []

for country, group in df.groupby(country_col):
    
    # Make quantile stratification key within this country’s slice
    group = group.copy()
    group['strata'] = make_strata_key(group, numeric_cols)
    
    # First 10 percent
    g_train_temp, g_holdA = train_test_split(
        group,
        test_size=0.10,
        stratify=group['strata'],
        random_state=42
    )
    
    # Second 10 percent, again stratified
    g_train_final, g_holdB = train_test_split(
        g_train_temp,
        test_size=0.10 / 0.90,  
        stratify=g_train_temp['strata'],
        random_state=99
    )
    
    # Remove helper
    for subset in [g_train_final, g_holdA, g_holdB]:
        subset.drop(columns=['strata'], inplace=True)
    
    train_parts.append(g_train_final)
    holdoutA_parts.append(g_holdA)
    holdoutB_parts.append(g_holdB)

# merge back together
train_df = pd.concat(train_parts, ignore_index=True)
holdoutA_df = pd.concat(holdoutA_parts, ignore_index=True)
holdoutB_df = pd.concat(holdoutB_parts, ignore_index=True)
