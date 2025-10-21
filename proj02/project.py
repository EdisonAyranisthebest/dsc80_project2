# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pd.options.plotting.backend = 'plotly'

from IPython.display import display

# DSC 80 preferred styles
pio.templates["dsc80"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+dsc80"
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def clean_loans(loans):
    df = loans.copy()

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    df["term"] = df["term"].str.extract(r"(\d+)").astype("Int64")

    df["emp_title"] = df["emp_title"].astype(str).str.strip().str.lower()
    df.loc[df["emp_title"] == "rn", "emp_title"] = "registered nurse"

    df["term_end"] = df["issue_d"] + df["term"].apply(
        lambda m: pd.DateOffset(months=int(m)) if pd.notna(m) else pd.NaT
    )

    return df


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def correlations(df, pairs):
    results = {}
    for col1, col2 in pairs:
        corr_value = df[col1].corr(df[col2])
        results[f"r_{col1}_{col2}"] = corr_value

    return pd.Series(results)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_boxplot(loans):
    df = loans.copy()

    bins = [580, 670, 740, 800, 850]
    bin_order = ["[580, 670)", "[670, 740)", "[740, 800)", "[800, 850)"]
    fico_bin = pd.cut(
        df["fico_range_low"],
        bins=bins,
        right=False,            
        include_lowest=True
    )

    df = df.loc[fico_bin.notna(), :].assign(fico_bin=fico_bin.astype(str))

    term_vals = df["term"].astype("Int64")

    fig = px.box(
        df.assign(term=term_vals),
        x="fico_bin",
        y="int_rate",
        color="term",
        category_orders={"fico_bin": bin_order, "term": [36, 60]},
        color_discrete_map={36: "purple", 60: "gold"},
        title="Interest Rate vs. Credit Score",
        labels={
            "fico_bin": "Credit Score Range",
            "int_rate": "Interest Rate (%)",
            "term": "Loan Length (Months)"
        },
        points="outliers"
    )

    return fig


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def ps_test(loans, N):
    has_ps = loans["desc"].notna()

    rates = loans["int_rate"]
    valid = rates.notna()

    rates = rates[valid].to_numpy()
    labels = has_ps[valid].to_numpy()

    obs = rates[labels].mean() - rates[~labels].mean()

    diffs = np.empty(N, dtype=float)
    for i in range(N):
        shuffled = np.random.permutation(labels)
        diffs[i] = rates[shuffled].mean() - rates[~shuffled].mean()

    pval = float((diffs >= obs).mean())
    return pval
    
def missingness_mechanism():
    return 2
    
def argument_for_nmar():
    return ("Whether an applicant writes a personal statement likely depends on "
            "the unobserved content/strength of their own story (e.g., how compelling, "
            "private, or sensitive it is), so the probability of a statement being present "
            "depends on the value of the statement itself, not fully explained by other columns.")


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def tax_owed(income, brackets):
    if income <= 0:
        return 0.0

    total_tax = 0.0

    for i in range(len(brackets) - 1):
        rate, lower = brackets[i]
        next_lower = brackets[i + 1][1]

        if income > next_lower:
            total_tax += rate * (next_lower - lower)
        else:
            total_tax += rate * (income - lower)
            return total_tax

    top_rate, top_lower = brackets[-1]
    total_tax += top_rate * (income - top_lower)

    return total_tax


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_state_taxes(state_taxes_raw): 
    df = state_taxes_raw.copy()

    df = df.dropna(how="all")

    df["State"] = df["State"].astype("string")
    df["Rate"] = df["Rate"].astype("string")
    df["Lower Limit"] = df["Lower Limit"].astype("string")

    valid_state = df["State"].str.fullmatch(r"[A-Za-z.\-\s]+")
    df.loc[~valid_state.fillna(False), "State"] = pd.NA
    df["State"] = df["State"].ffill()

    rate_raw = df["Rate"].str.strip().str.lower()
    no_tax = rate_raw.eq("none")

    rate_num_str = rate_raw.where(~no_tax, "0")
    rate_num = pd.to_numeric(rate_num_str.str.replace("%", "", regex=False), errors="coerce")
    rate_prop = rate_num.div(100)
    rate_prop = rate_prop.apply(lambda v: round(v, 2) if pd.notna(v) else v)

    df["Rate"] = rate_prop.astype(float)

    ll = (
        df["Lower Limit"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    ll_num = pd.to_numeric(ll, errors="coerce").fillna(0)
    ll_num = ll_num.where(~no_tax, 0)
    df["Lower Limit"] = ll_num.astype(int)

    df = df[df["Rate"].notna()]

    return df[["State", "Rate", "Lower Limit"]]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def state_brackets(state_taxes):
    ordered = state_taxes.sort_values(["State", "Lower Limit"], kind="mergesort")

    blist = (
        ordered.groupby("State", sort=False)[["Rate", "Lower Limit"]]
        .apply(lambda g: list(zip(g["Rate"].astype(float), g["Lower Limit"].astype(int))))
        .to_frame(name="bracket_list")
    )

    return blist
    
def combine_loans_and_state_taxes(loans, state_taxes):
    # Start by loading in the JSON file.
    # state_mapping is a dictionary; use it!
    import json
    state_mapping_path = Path('data') / 'state_mapping.json'
    with open(state_mapping_path, 'r') as f:
        state_mapping = json.load(f)
        
    # Now it's your turn:
    sb = state_brackets(state_taxes).reset_index()  # columns: ['State', 'bracket_list']

    sb["State"] = sb["State"].map(state_mapping)

    sb = sb.dropna(subset=["State"])

    out = loans.copy()
    if "addr_state" in out.columns and "State" not in out.columns:
        out = out.rename(columns={"addr_state": "State"})

    out = out.merge(sb, on="State", how="left")

    return out


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def find_disposable_income(loans_with_state_taxes):
    FEDERAL_BRACKETS = [
     (0.1, 0), 
     (0.12, 11000), 
     (0.22, 44725), 
     (0.24, 95375), 
     (0.32, 182100),
     (0.35, 231251),
     (0.37, 578125)
    ]
    df = loans_with_state_taxes.copy()

    df["federal_tax_owed"] = df["annual_inc"].apply(
        lambda inc: round(tax_owed(inc, FEDERAL_BRACKETS), 2)
        if pd.notna(inc)
        else np.nan
    )

    df["state_tax_owed"] = df.apply(
        lambda row: round(tax_owed(row["annual_inc"], row["bracket_list"]), 2)
        if (pd.notna(row["annual_inc"]) and isinstance(row["bracket_list"], list))
        else np.nan,
        axis=1
    )

    df["disposable_income"] = (
        df["annual_inc"] - df["federal_tax_owed"] - df["state_tax_owed"]
    )

    return df


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def aggregate_and_combine(loans, keywords, quantitative_column, categorical_column):
    df = loans.copy()

    row_order = ["MORTGAGE", "OWN", "RENT", "Overall"]
    df = df[df[categorical_column] != "ANY"]

    out_cols = {}
    for kw in keywords:
        mask = df["emp_title"].str.contains(kw, na=False)

        per_cat = (
            df.loc[mask]
              .groupby(categorical_column, dropna=True)[quantitative_column]
              .mean()
        )

        overall = pd.Series(
            {"Overall": df.loc[mask, quantitative_column].mean()}
        )

        s = pd.concat([per_cat, overall], axis=0).reindex(row_order)
        out_cols[f"{kw}_mean_{quantitative_column}"] = s

    result = pd.concat(out_cols, axis=1)
    result.columns = result.columns.get_level_values(0)
    return result


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def exists_paradox(loans, keywords, quantitative_column, categorical_column):
    
    tbl = aggregate_and_combine(loans, keywords, quantitative_column, categorical_column)
    colA = f"{keywords[0]}_mean_{quantitative_column}"
    colB = f"{keywords[1]}_mean_{quantitative_column}"
    head_ok = (tbl.iloc[:-1][[colA, colB]].dropna().eval(f"`{colA}` > `{colB}`")).all()
    overall_flip = tbl.iloc[-1][colA] < tbl.iloc[-1][colB]
    return bool(head_ok and overall_flip)
    
def paradox_example(loans):
    return {
        'loans': loans,
        'keywords': ['teacher', 'manager'],
        'quantitative_column': 'loan_amnt',
        'categorical_column': 'purpose'
    }
