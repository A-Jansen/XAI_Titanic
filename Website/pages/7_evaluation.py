import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.switch_page_button import switch_page
from oocsi_source import OOCSI
import datetime
from datetime import datetime

header1, header2, header3 = st.columns([1,4,1])
image1, image2, image3 = st.columns([1,50,1])
body1, body2, body3 =st.columns([1,50,1])

def record_page_start_time():
    global page_start_time
    page_start_time = datetime.now()

# Function to record page duration and send to Data Foundry
def record_page_duration_and_send():
    current_page_title = st.session_state.current_page_title
    if page_start_time:
        page_end_time = datetime.now()
        page_duration = page_end_time - page_start_time
        st.write(f"Time spent on {current_page_title}: {page_duration}")
        
        # Send data to Data Foundry via OOCSI
        data = {
            "page_name": current_page_title,
            "duration_seconds": page_duration.total_seconds()
        }
        st.session_state.oocsi.send('Time_XAI', data)

st.session_state.current_page_title = "Final Page"
page_start_time = None
record_page_start_time()


with header2:
    st.title("Comparing the different methods")
    st.markdown("This is the final section of this experiment, please rate and compare the different methods")


with image2:
    st.image('https://github.com/A-Jansen/XAI_Titanic/blob/main/Website/assets/images/overview%20methods.png?raw=true')

with body2:
    with st.form("my_form"):
        st.markdown("**As a final evaluation, please rate the different types of explanations (0-10). This is a general grade that you you would give to the different explanation methods.**")
        shap = st.slider('SHAP', 0, 10)
        dt = st.slider('Decision tree', 0, 10)
        counterfactual  = st.slider("Counterfactual", 0, 10)
        visualmap = st.slider("Visual map", 0, 10)
        favourite = st.radio("**What was your favourite type of epxlanation?**", ('SHAP', 'Decision tree', 'Counterfactual', 'Visual map'))
        why = st.text_area('**Please explain why**', "")
        # Every form must have a submit button.

        submitted = st.form_submit_button("Submit")

        if submitted:
            if page_start_time:
                record_page_duration_and_send()
            # record_page_start_time()
            st.session_state.oocsi.send('XAI_endcomparison', {
                'participant_ID': st.session_state.participantID,
                'shap': shap,
                'decisiontree': dt,
                'counterfactual': counterfactual,
                'visualmap': visualmap,
                'favourite': favourite,
                'why': why
            })
            st.balloons()
            switch_page('thankyou')