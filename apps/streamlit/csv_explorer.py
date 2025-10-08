import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="CSV Explorer", layout="wide")
st.title("CSV Explorer (Upload → Plot)")

file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.dataframe(df.head(50), use_container_width=True)
    cols = df.columns.tolist()
    if not cols:
        st.warning("Uploaded file contains no columns to plot.")
        st.stop()

    x = st.selectbox("X", cols, index=0)
    y_default = 1 if len(cols) > 1 else 0
    y = st.selectbox("Y", cols, index=y_default)
    color = st.selectbox("Color (optional)", [None] + cols)
    kind = st.selectbox("Chart", ["scatter", "line", "histogram"])
    if kind == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, trendline="ols")
    elif kind == "line":
        fig = px.line(df, x=x, y=y, color=color)
    else:
        fig = px.histogram(df, x=x, color=color)
    st.plotly_chart(fig, use_container_width=True)

# streamlit run apps\streamlit\csv_explorer.py
