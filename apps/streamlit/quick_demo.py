import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Streamlit Quick Demo", layout="wide")
st.title("Streamlit + Plotly")

df = pd.DataFrame({"x": range(50), "y": [i**0.5 for i in range(50)]})
x = st.selectbox("X", df.columns, index=0)
y_index = 1 if len(df.columns) > 1 else 0
y = st.selectbox("Y", df.columns, index=y_index)
fig = px.scatter(df, x=x, y=y, trendline="ols", title="Interactive scatter")
st.plotly_chart(fig, use_container_width=True)

# streamlit run apps\streamlit\quick_demo.py
