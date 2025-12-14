import streamlit as st

st.set_page_config(page_title="ML Project Report", layout="wide")

# Title and description
st.title("📘 ML Project Report")
st.write("Below is the full README for this project.")

# Load and display README
with open("README.md", "r", encoding="utf-8") as f:
    readme_text = f.read()

st.markdown(readme_text, unsafe_allow_html=True)
