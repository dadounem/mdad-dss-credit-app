from st_pages import Page, show_pages, add_page_title
import streamlit as st

st.set_page_config(
    page_title="My Credit App",
    page_icon="👋",
)

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("pages/home.py", "Home", "🏠"),
        Page("pages/predict.py", "Predict"),
        Page("pages/insights.py", "Insights"),
        Page("pages/train.py", "Train")
    ]
)