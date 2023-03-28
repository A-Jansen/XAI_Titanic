import streamlit as st
import streamlit.components.v1 as components


header = st.container()



with header:
    st.title("The last questions")
    with st.form("my_form"):
        st.write("Inside the form")
        slider_val = st.slider("Form slider")
        checkbox_val = st.checkbox("Form checkbox")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("slider", slider_val, "checkbox", checkbox_val)
            st.balloons()
    # Execute your app
    # embed streamlit docs in a streamlit app


