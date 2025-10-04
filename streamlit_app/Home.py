import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="College Football Playoff Dashboard", layout="wide")

# --- Top banner ---
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="background-color:#002D62; padding:4px 10px; border-radius:10px; margin-bottom:10px;">
        <h1 style="color:white; text-align:center;">🏈 College Football Playoff Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)