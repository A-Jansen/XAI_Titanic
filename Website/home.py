"""
These are the meta data and instructions
Author: Anniek Jansen
Course: Human AI interaction

Instructions:
(For Anniek: conda activate py (activate local environment where the right packages are installed))
1. Pip install streamlit, oocsi in your python environment
2. Save this file somewhere on your computer
3. In the command line, cd to where your file is: "cd/...../folder

4. To run it: streamlit run home.py 
5. Click on the link it provides you
6. You need to click sometimes rerun in the website


To hide menu: copy paste this in config.toml
[ui]
hideSidebarNav = true

"""


import streamlit as st
import pandas as pd
# from oocsi import __init__ as OOCSI
from oocsi_source import OOCSI
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import requests
import json


# Check if 'participantID' already exists in session_state
# If not, then initialize it
if 'participantID' not in st.session_state:
    st.session_state.participantID= "P" + uuid4().__str__().replace('-', '')[0:10]
    st.session_state.profiles =['SHAP', 'DecisionTree', 'counterfactual', 'visualMap']

if 'oocsi' not in st.session_state:
    st.session_state.oocsi = OOCSI('','oocsi.id.tue.nl')


if 'URL' not in st.session_state:
    # API endpoint
    st.session_state.URL = "https://data.id.tue.nl/datasets/ts/record/7810/V0NLMVNwQmNLQUMrYlZVRi9kaUx2ejNhbUhtNzMrTURPS0JaZ2hxc213TT0="
    st.session_state.URLconsent="https://data.id.tue.nl/datasets/ts/record/7813/aDc4MzNpa2Q1RXQ4ZmI3MS85S2ozVUhkMitEY25GV3FBM3VkaEZ5Rm9uaz0="
# st.write(st.session_state.participantID)


header = st.container()
consent_form = st.container()

with header:
    st.title("Consent form")
    st.write("For debugging:")
    st.write(st.session_state.participantID)

with consent_form:
    st.markdown("A whole lot of text and options")
    agree = st.checkbox('I agree')
    if agree:
        st.session_state.oocsi.send('EngD_HAII_consent', {
            'participant_ID': st.session_state.participantID,
            'consent': 'yes'
            })
        
        st.write('Great!')
        if st.button("Next page"):
            switch_page("explanationpage")
