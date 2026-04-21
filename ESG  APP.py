import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("China Listed Companies ESG Evaluator")
st.subheader("ACC102 Track4 Data Product")

data = pd.DataFrame({
    "Company": ["Ping An", "China Mobile", "Alibaba", "Tencent", "BYD", "CATL", "ICBC", "Midea"],
    "E": [72,85,68,75,92,88,65,78],
    "S": [80,76,70,79,85,82,69,74],
    "G": [85,88,65,82,78,80,90,76]
})

data["Total"] = (data.E + data.S + data.G)/3

company = st.selectbox("Choose a company", data.Company)
d = data[data.Company == company].iloc[0]

st.metric("Total ESG Score", round(d.Total,1))

fig, ax = plt.subplots()
ax.bar(["E","S","G"], [d.E, d.S, d.G], color=["green","blue","orange"])
ax.set_ylim(0,100)
st.pyplot(fig)

st.dataframe(data.sort_values("Total", ascending=False))
