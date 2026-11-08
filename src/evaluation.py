import numpy as np
import pandas as pd
from scipy import stats


def cluster_sizes(labels):
    s = pd.Series(labels).value_counts().sort_index()
    return pd.DataFrame({"n": s, "%": (100 * s / s.sum()).round(1)})


def eta_squared(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    ss_between = sum(
        (labels == c).sum() * (values[labels == c].mean() - grand) ** 2 for c in np.unique(labels)
    )
    return ss_between / ss_total


def cramers_v(a, b):
    table = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


def vif_table(df):
    from sklearn.linear_model import LinearRegression

    out = {}
    for c in df.columns:
        others = df.drop(columns=c)
        r2 = LinearRegression().fit(others, df[c]).score(others, df[c])
        out[c] = 1 / (1 - r2)
    return pd.Series(out, name="VIF")
