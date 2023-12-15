import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.switch_page_button import switch_page
from oocsi_source import OOCSI
import datetime
from datetime import datetime

header1, header2, header3 = st.columns([1, 4, 1])
image1, image2, image3 = st.columns([1, 50, 1])
body1, body2, body3 = st.columns([1, 4, 1])


def check_input_length(text):
    words = text.split()
    word_count = len(words)
    return word_count


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
            "duration_seconds": page_duration.total_seconds(),
            'participant_ID': st.session_state.participantID
        }
        st.session_state.oocsi.send('Time_XAI', data)


st.session_state.current_page_title = "Final Page"
page_start_time = None
record_page_start_time()


# functions for sending the data via oocsi
def record_why1():
    st.session_state.oocsi.send('XAI_endcomparison_why1', {
        'participant_ID': st.session_state.participantID,
        'why_1': why,
    })


def record_why2():
    st.session_state.oocsi.send('XAI_endcomparison_why2', {
        'participant_ID': st.session_state.participantID,
        'why_2': why_2,
    })


def record_why3():
    st.session_state.oocsi.send('XAI_endcomparison_why3', {
        'participant_ID': st.session_state.participantID,
        'why_3': why_3,
    })


def record_why4():
    st.session_state.oocsi.send('XAI_endcomparison_why4', {
        'participant_ID': st.session_state.participantID,
        'why_4': why_4,
    })


def record_other():
    st.session_state.oocsi.send('XAI_endcomparison_other', {
        'participant_ID': st.session_state.participantID,
        'shap': shap,
        'decisiontree': dt,
        'counterfactual': counterfactual,
        'visualmap': visualmap,
        'favourite': favourite,
    })


with header2:
    st.title("Comparing the different methods")
    st.markdown("In this section, we ask you to compare and evaluate the four XAI methods you have just seen. If you need a reminder what they were, please look at the picture.")


with image2:
    st.image('https://github.com/A-Jansen/XAI_Titanic/blob/main/nonexperts/assets/images/overview%20methods.png?raw=true')

with body2:
    with st.form("my_form"):
        st.markdown("**As a final evaluation, please rate the different types of explanations (0-10). This is a general grade that you would give to the different explanation methods.**")
        shap = st.slider('SHAP', 0, 10)
        dt = st.slider('Decision tree', 0, 10)
        counterfactual = st.slider("Counterfactual", 0, 10)
        visualmap = st.slider("Visual map", 0, 10)
        favourite = st.radio("**What was your favourite type of explanation?**",
                             ('SHAP', 'Decision tree', 'Counterfactual', 'Visual map'))
        why = st.text_area(
            '**Please explain what you liked about your favourite XAI method and why**', "")
        why_2 = st.text_area(
            '**Please briefly mention why you disliked the other three XAI methods.  Use the format: Name of method (1 / 2 / 3): Reason to dislike (1 / 2 / 3)**')
        why_3 = st.text_area(
            '**Please explain how your favourite XAI method helped you to understand the prediction of the ML model**')
        why_4 = st.text_area(
            '**Optional, If you have any remarks regarding the different methods you could input them here**')
        # Every form must have a submit button.
        word_count_1 = check_input_length(why)
        word_count_2 = check_input_length(why_2)
        word_count_3 = check_input_length(why_3)
        word_count_4 = check_input_length(why_4)

        submitted = st.form_submit_button("Submit")

        if submitted:
            if word_count_1 < 7:
                st.warning(
                    'Please explain more extensively your answer (+7 words)')
            elif word_count_2 < 7:
                st.warning(
                    'Please explain more extensively your answer (+7 words)')
            elif word_count_3 < 7:
                st.warning(
                    'Please explain more extensively your answer (+7 words)')
            elif word_count_4 < 7:
                st.warning(
                    'Please explain more extensively your answer (+7 words)')
            else:
                st.success('Thank you!')
                if page_start_time:
                    record_page_duration_and_send()
                # record_page_start_time()
                    record_why1()
                    record_why2()
                    record_why3()
                    record_why4()
                    record_other()

                    switch_page('evaluation')
