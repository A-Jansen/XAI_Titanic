import streamlit as st
from uuid import uuid4
from streamlit_extras.switch_page_button import switch_page
import random
import pandas as pd
import datetime
import xgboost as xgb
import copy
from PIL import Image
from datetime import datetime
import numpy as np

header1, header2, header3 = st.columns([1,2,1])
body1, body2, body3 =st.columns([1,2,1])
footer1, footer2, footer3 =st.columns([1,2,1])



with body2:
    st.subheader("Without consent you cannot participate in this study, please click on the link to go back to Prolific")
    # code for the non-experts
    st.write("https://app.prolific.com/submissions/complete?cc=C16N6GP4")


