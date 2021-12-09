import pandas as pd
import streamlit as st

data_frame = pd.read_csv("raw_data/reviews.csv")
map_data = data_frame[["latitude", "longitude"]]

st.map(map_data)
