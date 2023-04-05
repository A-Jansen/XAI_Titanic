import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.switch_page_button import switch_page
from oocsi_source import OOCSI

header1, header2, header3 = st.columns([1,4,1])
image1, image2, image3 = st.columns([1,50,1])
body1, body2, body3 =st.columns([1,2,1])


with header2:
    st.title("Comparing the different methods")
    st.markdown("This is the final section of this experiment, please rate and compare the different methods")

with image2:
    st.image('https://github.com/A-Jansen/XAI_Titanic/blob/main/Website/assets/images/overview%20methods.png?raw=true')

with body2:
    with st.form("my_form"):
        st.write("As a final evaluation, please rate the different types of explanations (0-10). This is a general grade that you you would give to the different explanation methods.")


        shap = st.slider('SHAP', 0, 10)
        dt = st.slider('Decision tree', 0, 10)
        counterfactual  = st.slider("Counterfactual", 0, 10)
        visualmap = st.slider("Visual map", 0, 10)
        favourite = st.radio("What was your favourite type of epxlanation?", ('SHAP', 'Decision tree', 'Counterfactual', 'Visual map'))
        why = st.text_area('Please explain why', "")
        # Every form must have a submit button.

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.oocsi.send('EngD_HAII_comparison', {
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
    # Execute your app
    # embed streamlit docs in a streamlit app


