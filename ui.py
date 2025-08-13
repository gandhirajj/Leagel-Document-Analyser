import os
import streamlit as st


def inject_css():
	css_path = os.path.join("assets", "styles.css")
	if os.path.exists(css_path):
		with open(css_path, "r", encoding="utf-8") as f:
			st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
