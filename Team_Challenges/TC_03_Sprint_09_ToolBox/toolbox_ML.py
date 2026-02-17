import pandas as pd
import numpy as np
from scipy.stats import pearsonr, f_oneway, kruskal, normaltest
import seaborn as sns
import matplotlib.pyplot as plt


def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Saca un resumen rápido del DataFrame para ver cómo son las columnas.

    Argumentos:
    df (pd.DataFrame): DataFrame que quieres analizar.

    Retorna:
    pd.DataFrame: Una tabla resumen donde por cada columna te dice:
        - el tipo de dato
        - % de nulos
        - cuántos valores distintos tiene
        - % de cardinalidad (distintos / filas * 100)
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas.DataFrame")
        return None

    n = len(df)
    out = pd.DataFrame(index=df.columns)
    out["tipo"] = df.dtypes.astype(str)
    out["%_nulos"] = (df.isna().mean() * 100).round(2)
    out["valores_unicos"] = df.nunique(dropna=True)
    out["%_cardinalidad"] = ((out["valores_unicos"] / n) * 100).round(2) if n > 0 else 0.0
    return out.T


def tipifica_variables(df: pd.DataFrame, umbral_categoria: int, umbral_continua: float) -> pd.DataFrame:
    """
    Intenta adivinar el tipo de cada columna usando su cardinalidad.

    Argumentos:
    df (pd.DataFrame): DataFrame a revisar.
    umbral_categoria (int): Si una columna tiene menos únicos que este número, se considera "Categórica".
    umbral_continua (float): Porcentaje (0-100). Si la % cardinalidad supera este valor, la numérica se considera "Continua".

    Retorna:
    pd.DataFrame: Tabla con el nombre de cada columna y el tipo sugerido
                  (Binaria / Categórica / Numerica Continua / Numerica Discreta).
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas.DataFrame")
        return None
    if not isinstance(umbral_categoria, int) or umbral_categoria < 2:
        print("Error: umbral_categoria debe ser int >= 2")
        return None
    if not (isinstance(umbral_continua, (int, float)) and 0 <= umbral_continua <= 100):
        print("Error: umbral_continua debe ser un porcentaje entre 0 y 100")
        return None

    n = len(df)
    rows = []
    for col in df.columns:
        card = df[col].nunique(dropna=True)
        pct_card = (card / n) * 100 if n > 0 else 0

        if card == 2:
            t = "Binaria"
        elif card < umbral_categoria:
            t = "Categórica"
        else:
            t = "Numerica Continua" if pct_card >= umbral_continua else "Numerica Discreta"

        rows.append({"nombre_variable": col, "tipo_sugerido": t})

    return pd.DataFrame(rows)


def get_features_num_regression(
    df: pd.DataFrame,
    target_col: str,
    umbral_corr: float,
    pvalue: float | None = None
) -> list | None:
    """
    Devuelve una lista de columnas numéricas que se relacionan con el target (para regresión)
    mirando la correlación de Pearson.

    Argumentos:
    df (pd.DataFrame): DataFrame con target y el resto de columnas.
    target_col (str): Nombre de la columna objetivo (tiene que ser numérica).
    umbral_corr (float): Umbral (0-1). Si |corr| es mayor que esto, se queda.
    pvalue (float | None): Si se pasa, además se pide que sea estadísticamente significativo (p <= pvalue).

    Retorna:
    list | None: Lista con nombres de columnas que pasan el filtro.
                 Devuelve None si hay algún parámetro mal.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas.DataFrame")
        return None
    if target_col not in df.columns:
        print("Error: target_col no existe en df")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col debe ser numérica")
        return None
    if not (isinstance(umbral_corr, (int, float)) and 0 <= abs(umbral_corr) <= 1):
        print("Error: umbral_corr debe estar entre 0 y 1")
        return None
    if pvalue is not None and not (isinstance(pvalue, (int, float)) and 0 < pvalue <= 1):
        print("Error: pvalue debe estar en (0, 1]")
        return None

    target = df[target_col]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    selected = []
    for c in num_cols:
        x = df[c]
        mask = target.notna() & x.notna()
        if mask.sum() < 3:
            continue
        r, pv = pearsonr(x[mask], target[mask])
        if abs(r) > umbral_corr and (pvalue is None or pv <= pvalue):
            selected.append(c)

    return selected


def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str,
    columns: list = [],
    umbral_corr: float = 0,
    pvalue: float | None = None
) -> list | None:
    """
    Hace gráficos (pairplot) para ver la relación entre el target y varias columnas numéricas.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre del target (numérico).
    columns (list): Lista de columnas numéricas a probar. Si está vacía, prueba todas las numéricas menos el target.
    umbral_corr (float): Umbral de correlación (0-1) para filtrar.
    pvalue (float | None): Si se pasa, también filtra por p-valor.

    Retorna:
    list | None: Lista final de columnas numéricas que cumplen el filtro y que se han usado para graficar.
                 Devuelve None si hay un error en los parámetros.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas.DataFrame")
        return None
    if target_col not in df.columns:
        print("Error: target_col no existe en df")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col debe ser numérica")
        return None

    if columns is None:
        columns = []

    if len(columns) == 0:
        cand = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    else:
        cand = []
        for c in columns:
            if c in df.columns and np.issubdtype(df[c].dtype, np.number) and c != target_col:
                cand.append(c)

    if len(cand) == 0:
        print("No hay variables numéricas candidatas.")
        return []

    tmp = df[[target_col] + cand].copy()
    selected = get_features_num_regression(tmp, target_col, umbral_corr, pvalue)
    if selected is None:
        return None
    if len(selected) == 0:
        print("Ninguna variable numérica cumple el criterio.")
        return []

    block_size = 4
    for i in range(0, len(selected), block_size):
        block = selected[i:i + block_size]
        data = df[[target_col] + block].dropna()
        if len(data) < 3:
            continue
        sns.pairplot(data)
        plt.show()

    return selected


def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list | None:
    """
    Busca columnas categóricas que parezcan tener relación con un target numérico.
    (O sea, que según la categoría el target cambia bastante).

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre del target (tiene que ser numérico).
    pvalue (float): Umbral de significancia. Si p <= pvalue, se considera que hay relación.

    Retorna:
    list | None: Lista con las columnas categóricas que pasan el test.
                 Devuelve None si algo está mal en los parámetros.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas.DataFrame")
        return None
    if target_col not in df.columns:
        print("Error: target_col no existe en df")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col debe ser numérica")
        return None
    if not (isinstance(pvalue, (int, float)) and 0 < pvalue <= 1):
        print("Error: pvalue debe estar en (0, 1]")
        return None

    cat_cols = [c for c in df.columns if c != target_col and not np.issubdtype(df[c].dtype, np.number)]

    selected = []
    y = df[target_col]

    for c in cat_cols:
        sub = df[[c, target_col]].dropna()
        if sub[c].nunique() < 2:
            continue

        groups = [sub.loc[sub[c] == lvl, target_col].values for lvl in sub[c].unique()]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue

        normal_ok = True
        for g in groups:
            if len(g) >= 8:
                try:
                    _, pv_norm = normaltest(g)
                    if pv_norm <= 0.05:
                        normal_ok = False
                        break
                except Exception:
                    normal_ok = False
                    break
            else:
                normal_ok = False
                break

        try:
            if normal_ok:
                _, pv = f_oneway(*groups)
            else:
                _, pv = kruskal(*groups)
        except Exception:
            continue

        if pv <= pvalue:
            selected.append(c)

    return selected


def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    columns: list = [],
    pvalue: float = 0.05,
    with_individual_plot: bool = False
) -> list | None:
    """
    Dibuja histogramas del target separando por categorías para las columnas que salen significativas.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre del target (numérico).
    columns (list): Columnas categóricas a probar. Si está vacío, usa todas las categóricas del df.
    pvalue (float): Umbral de significancia para el test.
    with_individual_plot (bool): Si True, también hace un gráfico por cada categoría (puede sacar muchos).

    Retorna:
    list | None: Lista final de columnas categóricas que han pasado el filtro (y que se han graficado).
                 Devuelve None si hay un error en los parámetros.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas.DataFrame")
        return None
    if target_col not in df.columns:
        print("Error: target_col no existe en df")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: target_col debe ser numérica")
        return None

    if columns is None:
        columns = []

    if len(columns) == 0:
        cand = [c for c in df.columns if c != target_col and not np.issubdtype(df[c].dtype, np.number)]
    else:
        cand = [c for c in columns if c in df.columns and c != target_col and not np.issubdtype(df[c].dtype, np.number)]

    if len(cand) == 0:
        print("No hay variables categóricas candidatas.")
        return []

    tmp = df[[target_col] + cand].copy()
    selected = get_features_cat_regression(tmp, target_col, pvalue)
    if selected is None:
        return None
    if len(selected) == 0:
        print("Ninguna variable categórica cumple el criterio.")
        return []

    for c in selected:
        data = df[[c, target_col]].dropna()
        plt.figure()
        sns.histplot(data=data, x=target_col, hue=c, element="step", stat="density", common_norm=False)
        plt.title(f"{target_col} por {c}")
        plt.show()

        if with_individual_plot:
            for lvl in data[c].unique():
                plt.figure()
                sns.histplot(data=data[data[c] == lvl], x=target_col, stat="density")
                plt.title(f"{target_col} | {c} = {lvl}")
                plt.show()

    return selected
