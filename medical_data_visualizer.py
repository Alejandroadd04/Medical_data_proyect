import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")
# 2
df['IMC'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['IMC'] > 25).astype(int)
# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
# 4

def draw_cat_plot():
    df_cat = df[['cardio', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']]

    df_cat_melted = pd.melt(
    df_cat,
    id_vars=['cardio'],
    value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    sns.set(style="whitegrid")

    cat_plot = sns.catplot(
    data=df_cat_melted,
    x='variable',
    hue='value',
    col='cardio',
    kind='count'
    )

    cat_plot.set_axis_labels("variable", "total")

    fig = cat_plot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ].copy()
    # 12
    df_heat = df_heat.drop(columns=['IMC'])
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))


    # 14
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})

    # 16
    fig.savefig('heatmap.png')
    return fig
