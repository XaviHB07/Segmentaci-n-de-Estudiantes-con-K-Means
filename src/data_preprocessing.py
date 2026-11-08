from pathlib import Path

import pandas as pd

DATA_FILE = "student_habits.csv"

ID_COL = "student_id"
CLUSTER_FEATURES = [
    "study_hours",
    "sleep_hours",
    "social_hours",
    "physical_activity_hours",
    "screen_time_hours",
    "extracurricular_hours",
]
EXTERNAL_COLS = ["gpa", "stress_score", "stress_level"]
DEMOGRAPHIC_COLS = ["age", "field_of_study"]
STRESS_ORDER = ["Low", "Moderate", "High"]


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for folder in [start, *start.parents]:
        if (folder / "data" / "raw").is_dir():
            return folder
    raise FileNotFoundError("No se encontró data/raw. Ejecuta el notebook dentro del repositorio.")


def load_data(root=None):
    root = Path(root) if root else find_project_root()
    path = root / "data" / "raw" / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Genera los datos con: python src/generate_data.py")
    return pd.read_csv(path)


def quality_summary(df):
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "nulos_%": (100 * df.isna().mean()).round(2),
        "unicos": df.nunique(),
    })
    num = df.select_dtypes("number")
    out["min"] = num.min()
    out["max"] = num.max()
    return out


def iqr_outliers(df, cols, k=1.5):
    flags = {}
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        flags[c] = (df[c] < q1 - k * iqr) | (df[c] > q3 + k * iqr)
    return pd.DataFrame(flags)


def near_constant(df, cols):
    return pd.Series({c: df[c].value_counts(normalize=True).iloc[0] for c in cols})
