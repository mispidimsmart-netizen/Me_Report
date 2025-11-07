
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import requests
from io import StringIO
from urllib.parse import quote

PUBLISHED_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRkcagLu_YrYgQxmsO3DnHn90kqALkw9uDByX7UBNRUjaFKKQdE3V-6fm5ZcKGk_A/pub?gid=2143275417&single=true&output=csv"""
BASE_SHEET = "Baseline Info"

def col_letter_to_pos(s: str) -> int:
    s = re.sub(r'[^A-Za-z]', '', s)
    v = 0
    for ch in s.upper():
        v = v * 26 + (ord(ch) - 64)
    return v

BRANCH_COL_POS = col_letter_to_pos("G")
LOAN_TYPE_POS  = col_letter_to_pos("AN")
AMOUNT_POS     = col_letter_to_pos("AQ")

st.set_page_config(page_title="PIDIM Foundation — SMART Project Reports", layout="wide")

hdr_l, hdr_r = st.columns([0.12, 0.88])
with hdr_l:
    st.image("assets/logo.png", width=72)
with hdr_r:
    st.markdown("<div style='font-size:34px; font-weight:800; color:#16a34a; line-height:1;'>PIDIM Foundation</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:16px; color:#334155; margin-top:4px;'>Sustainable Microenterprise and Resilient Transformation (SMART) Project</div>", unsafe_allow_html=True)

tb1, tb2, tb3 = st.columns([0.18, 0.14, 0.68])
with tb1:
    refresh = st.button("🔄 Refresh data")
with tb2:
    fast_mode = st.toggle("⚡ Fast mode", value=True, help="Turn off heavy styling for big speed-ups")
with tb3:
    st.write("")

st.markdown(r"""
<style>
  .block-container {padding-top: 0.6rem; padding-bottom: 1.2rem;}
  .card { background:#ffffff; border:1px solid #eaecef; border-radius:14px; box-shadow:0 1px 3px rgba(0,0,0,0.06); padding:14px 16px; margin-bottom:18px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "ME-Reports/1.0"})
    return s

@st.cache_data(ttl=900, max_entries=3, show_spinner=False)
def fetch_csv(url: str) -> str:
    s = get_session()
    r = s.get(url, timeout=20)
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=900, max_entries=3, show_spinner=False)
def parse_df(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df

if refresh:
    fetch_csv.clear(); parse_df.clear()
    st.toast("Cache cleared. Reloading...", icon="🔁")
    st.experimental_rerun()

csv_text = fetch_csv(PUBLISHED_CSV_URL)
df = parse_df(csv_text)
if df is None or df.empty:
    st.error("Sheet is empty or not reachable."); st.stop()

with st.expander("📂 View Raw Data (from Google Sheet)", expanded=False):
    st.dataframe(df, use_container_width=True)

def _get_by_pos(df: pd.DataFrame, pos_1based: int) -> str:
    if not len(df.columns): return None
    idx0 = max(0, min(len(df.columns)-1, pos_1based-1)); return df.columns[idx0]

def _best_guess_col(df: pd.DataFrame, keywords):
    text_cols = [c for c in df.columns if df[c].dtype == object]; score = []
    for c in text_cols:
        vals = pd.Series(df[c].dropna().astype(str).unique()).str.lower(); hits = 0
        for kw in keywords: hits += (vals.str.contains(kw.lower(), na=False)).sum()
        score.append((hits, c))
    score.sort(reverse=True); 
    if score and score[0][0] > 0: return score[0][1]
    return text_cols[0] if text_cols else df.columns[0]

@st.cache_data(ttl=900, show_spinner=False)
def compute_branch_loan(df_in, branch_col, loan_type_col, amount_col):
    work = df_in[[branch_col, loan_type_col, amount_col]].copy()
    work[branch_col] = work[branch_col].astype(str).str.strip()
    work[loan_type_col] = work[loan_type_col].astype(str).str.strip()
    work["_Amount_"] = pd.to_numeric(work[amount_col], errors="coerce")
    hdr_like_branch_vals = {"branch","branch name","branchname"}
    hdr_like_type_vals = {"enterprise/non-enterprise","enterprise - non-enterprise","enterprise non-enterprise"}
    work = work[~work[branch_col].str.lower().isin(hdr_like_branch_vals)]
    work = work[~work[loan_type_col].str.lower().isin(hdr_like_type_vals)]
    work = work[work[branch_col].notna() & (work[branch_col].astype(str).str.strip().str.lower().isin(["","nan","none"]) == False)]
    def _norm_type(x: str):
        x = (x or "").strip().lower(); x = re.sub(r'\s+', ' ', x)
        if "non" in x and "enterprise" in x: return "Non-Enterprise"
        if "enterprise" in x: return "Enterprise"
        return x.title() if x else x
    work["_TypeNorm_"] = work[loan_type_col].apply(_norm_type)
    agg = (work.groupby([branch_col, "_TypeNorm_"], dropna=False)
              .agg(**{"# of Loan":("_TypeNorm_","count"), "Amount of Loan":("_Amount_","sum")})
              .reset_index()
              .rename(columns={branch_col:"Branch Name","_TypeNorm_":"Types of Loan"}))
    return agg

@st.cache_data(ttl=900, show_spinner=False)
def compute_poultry(df_in, branch_col, activity_col, birds_col):
    tmp = df_in.copy()
    for c in [branch_col, activity_col, birds_col]: tmp[c] = tmp[c].astype(str).str.strip()
    tmp["_BirdsNum_"] = pd.to_numeric(tmp[birds_col], errors="coerce")
    def _count_and_sum(df_in2, b, keyword):
        d = df_in2[df_in2[activity_col].str.lower().str.contains(keyword, na=False)].copy()
        cnt = d.groupby(b, dropna=False).size().reset_index(name="count")
        birds = d.groupby(b, dropna=False)["_BirdsNum_"].sum(min_count=1).reset_index(name="birds")
        return cnt.merge(birds, on=b, how="outer")
    layer_stats = _count_and_sum(tmp, branch_col, "layer")
    broiler_stats = _count_and_sum(tmp, branch_col, "broiler")
    base_branches = pd.DataFrame({branch_col: sorted(tmp[branch_col].dropna().unique(), key=lambda x: str(x))})
    out2 = base_branches.merge(layer_stats.rename(columns={"count":"Layer Rearing","birds":"Layer # of Birds"}), how="left", on=branch_col)
    out2 = out2.merge(broiler_stats.rename(columns={"count":"Broiler Rearing","birds":"Broiler # of Birds"}), how="left", on=branch_col)
    out2 = out2.fillna(0).rename(columns={branch_col:"Branch Name"})
    return out2

@st.cache_data(ttl=900, show_spinner=False)
def compute_farm_counts(df_in, branch_col, farmtype_col):
    tmpf = df_in.copy()
    for c in [farmtype_col, branch_col]: tmpf[c] = tmpf[c].astype(str).str.strip()
    def _count_type(df_in2, b, key):
        d = df_in2[df_in2[farmtype_col].str.lower().str.contains(key, na=False)]
        return d.groupby(b, dropna=False).size().reset_index(name="count")
    gen = _count_type(tmpf, branch_col, "general")
    mod = _count_type(tmpf, branch_col, "model")
    branches = pd.DataFrame({branch_col: sorted(tmpf[branch_col].dropna().unique(), key=lambda x: str(x))})
    rep3 = branches.merge(gen.rename(columns={"count":"General Farm"}), how="left", on=branch_col)
    rep3 = rep3.merge(mod.rename(columns={"count":"Model Farm"}), how="left", on=branch_col)
    rep3 = rep3.fillna(0).rename(columns={branch_col:"Branch Name"})
    return rep3

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Branch Wise Loan Disbursement")
    branch_col = _get_by_pos(df, BRANCH_COL_POS)
    loan_type_col = _get_by_pos(df, LOAN_TYPE_POS)
    amount_col = _get_by_pos(df, AMOUNT_POS)
    with st.expander("Check / Adjust detected columns", expanded=False):
        branch_col = st.selectbox("Branch column (default G)", options=list(df.columns), index=list(df.columns).index(branch_col))
        loan_type_col = st.selectbox("Loan Type column (default AN)", options=list(df.columns), index=list(df.columns).index(loan_type_col))
        amount_col = st.selectbox("Amount column (default AQ)", options=list(df.columns), index=list(df.columns).index(amount_col))
    agg = compute_branch_loan(df, branch_col, loan_type_col, amount_col)
    type_order = {"Enterprise":0, "Non-Enterprise":1}
    agg["_ord_"] = agg["Types of Loan"].map(type_order).fillna(99).astype(int)
    rows = []
    for b, g in agg.sort_values(["Branch Name","_ord_"]).groupby("Branch Name", sort=False):
        for _, r in g.iterrows():
            rows.append({"Branch Name": str(b), "Types of Loan": r["Types of Loan"], "# of Loan": int(r["# of Loan"]), "Amount of Loan": float(r["Amount of Loan"] or 0)})
        rows.append({"Branch Name": str(b) + " Total", "Types of Loan": "", "# of Loan": int(g["# of Loan"].sum()), "Amount of Loan": float(g["Amount of Loan"].sum())})
    if rows:
        _tmp = pd.DataFrame(rows)
        rows.append({"Branch Name": "Grand Total", "Types of Loan": "", "# of Loan": int(_tmp[_tmp["Types of Loan"] != ""]["# of Loan"].sum()), "Amount of Loan": float(_tmp[_tmp["Types of Loan"] != ""]["Amount of Loan"].sum())})
    out_df = pd.DataFrame(rows)
    def _is_bad_branch(val: str):
        v = str(val).strip().lower(); return (v in ["nan","none",""]) or v.endswith("nan total")
    out_df = out_df[~out_df["Branch Name"].apply(_is_bad_branch)].copy()
    out_df.insert(0, "ক্রমিক নং", range(1, len(out_df)+1))
    if fast_mode:
        show_df = out_df.copy()
        show_df["Amount of Loan"] = pd.to_numeric(show_df["Amount of Loan"], errors="coerce").fillna(0).round(0).astype(int)
        st.dataframe(show_df, use_container_width=True)
    else:
        def style_fixed1(df_in):
            df_show = df_in.copy()
            df_show["Amount of Loan"] = pd.to_numeric(df_show["Amount of Loan"], errors="coerce").fillna(0).round(0).astype(int).map(lambda x: f"{x:,}")
            styler = df_show.style.hide(axis="index")
            def _row_style(row):
                name = str(row.get("Branch Name",""))
                if name.endswith(" Total") and name != "Grand Total": return ["background-color:#fffbe6; color:#000; font-weight:700"]*len(row)
                if name == "Grand Total": return ["background-color:#ffe59a; color:#000; font-weight:700"]*len(row)
                return [""]*len(row)
            styler = styler.apply(_row_style, axis=1)
            return styler
        st.dataframe(style_fixed1(out_df), use_container_width=True)
    try:
        chart_base = out_df[(~out_df["Branch Name"].str.endswith(" Total")) & (out_df["Branch Name"]!="Grand Total") & (out_df["Types of Loan"]!="")].copy()
        branch_order = sorted(chart_base["Branch Name"].unique())
        fig1 = px.bar(chart_base, x="Branch Name", y="Amount of Loan", color="Types of Loan", barmode="group", category_orders={"Branch Name": branch_order})
        fig1.update_layout(title_text="Amount of Loan by Branch & Type")
        try: fig1.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        except Exception: pass
        st.plotly_chart(fig1, use_container_width=True)
    except Exception: pass
    try:
        detail_mask = (out_df["Types of Loan"]!="") & (~out_df["Branch Name"].str.endswith(" Total")) & (out_df["Branch Name"]!="Grand Total")
        total_loanees = int(out_df.loc[detail_mask, "# of Loan"].sum())
        total_amount = float(out_df.loc[detail_mask, "Amount of Loan"].sum())
        summary_df = pd.DataFrame([{"Total Loanees": total_loanees, "Total Amount": total_amount}])
        st.dataframe(summary_df, use_container_width=False)
        st.download_button("⬇️ Download — Overall Summary (CSV)", data=summary_df.to_csv(index=False).encode("utf-8"), file_name="fixed_report_1_overall_summary.csv", mime="text/csv")
    except Exception: pass
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🐔 Types of Poultry Rearing")
    def _best_guess_birds_col(df0):
        cols = list(df0.columns); bird_like = [c for c in cols if ("bird" in str(c).lower())]
        num_with_bird = [c for c in bird_like if pd.api.types.is_numeric_dtype(df0[c])]
        if num_with_bird: return num_with_bird[0]
        for c in bird_like:
            try:
                if pd.to_numeric(df0[c], errors="coerce").notna().any(): return c
            except Exception: pass
        any_num = [c for c in cols if pd.api.types.is_numeric_dtype(df0[c])]
        return any_num[0] if any_num else cols[0]
    activity_col_guess = _best_guess_col(df, ["layer","broiler","layer rearing","broiler rearing"])
    birds_col_guess = _best_guess_birds_col(df)
    with st.expander("Check / Adjust detected columns (Poultry)", expanded=False):
        branch_col_p = st.selectbox("Branch column", options=list(df.columns), index=list(df.columns).index(_get_by_pos(df, BRANCH_COL_POS)))
        activity_col = st.selectbox("Activity column (Layer/Broiler)", options=list(df.columns), index=(list(df.columns).index(activity_col_guess) if activity_col_guess in df.columns else 0))
        birds_col = st.selectbox("# of Birds column", options=list(df.columns), index=(list(df.columns).index(birds_col_guess) if birds_col_guess in df.columns else 0))
    out2 = compute_poultry(df, branch_col_p, activity_col, birds_col)
    grand2 = pd.DataFrame({
        "Branch Name": ["Grand Total"],
        "Layer Rearing": [int(out2["Layer Rearing"].sum())],
        "Layer # of Birds": [int(out2["Layer # of Birds"].sum())],
        "Broiler Rearing": [int(out2["Broiler Rearing"].sum())],
        "Broiler # of Birds": [int(out2["Broiler # of Birds"].sum())],
    })
    out2_disp = pd.concat([out2, grand2], ignore_index=True)
    st.dataframe(out2_disp, use_container_width=True)
    st.download_button("⬇️ Download — Types of Poultry Rearing (CSV)", data=out2_disp.to_csv(index=False).encode("utf-8"), file_name="types_of_poultry_rearing.csv", mime="text/csv")
    st.markdown("#### Quick View")
    try:
        c1, c2 = st.columns(2)
        with c1:
            fig2a = px.bar(out2, x="Branch Name", y=["Layer Rearing","Broiler Rearing"], barmode="group", title="Poultry Rearing (Count) by Branch")
            try: fig2a.update_traces(texttemplate="%{value:,.0f}", textposition="outside")
            except Exception: pass
            st.plotly_chart(fig2a, use_container_width=True)
        with c2:
            fig2b = px.bar(out2, x="Branch Name", y=["Layer # of Birds","Broiler # of Birds"], barmode="group", title="# of Birds by Branch (Layer vs Broiler)")
            try: fig2b.update_traces(texttemplate="%{value:,.0f}", textposition="outside")
            except Exception: pass
            st.plotly_chart(fig2b, use_container_width=True)
    except Exception: pass
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🏷️ Model vs General Farm — Branch-wise Count")
farmtype_col_guess = _best_guess_col(df, ["general farm","model farm","general","model"])
with st.expander("Check / Adjust detected column (Farm Type)", expanded=False):
    farmtype_col = st.selectbox("Farm Type column (General/Model)", options=list(df.columns), index=(list(df.columns).index(farmtype_col_guess) if farmtype_col_guess in df.columns else 0))
    branch_col_f = st.selectbox("Branch column", options=list(df.columns), index=list(df.columns).index(_get_by_pos(df, BRANCH_COL_POS)))
rep3 = compute_farm_counts(df, branch_col_f, farmtype_col)
grand3 = pd.DataFrame({
    "Branch Name": ["Grand Total"],
    "General Farm": [int(rep3["General Farm"].sum())],
    "Model Farm": [int(rep3["Model Farm"].sum())],
})
rep3_disp = pd.concat([rep3, grand3], ignore_index=True)
st.dataframe(rep3_disp, use_container_width=True)
st.download_button("⬇️ Download — Model vs General Farm (CSV)", data=rep3_disp.to_csv(index=False).encode("utf-8"), file_name="model_vs_general_farm_counts.csv", mime="text/csv")
st.markdown("</div>", unsafe_allow_html=True)
