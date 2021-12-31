from requests.api import head
import streamlit as st
import time
import requests
import urllib.parse


def main():
    st.set_page_config(  # Alternate names: setup_page, page, layout
        # Can be "centered" or "wide". In the future also "dashboard", etc.
        layout="wide",
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        # String or None. Strings get appended with "• Streamlit".
        page_title="GPT-J Language Model",
        page_icon=None,  # String, anything supported by st.image, or None.
    )
    st.title("GPT-J Language Model")

    input = st.text_area("", max_chars=20000, height=150)

    response = None
    with st.form(key="inputs"):
        submit_button = st.form_submit_button(label="Generate!")

        if submit_button:

            params = {
                "text": input,
                "min_length": 40,
                "max_length": 340,
                "temperature": 0,
                "top_p": 0.9
            }

            query = requests.post(
                "http://localhost:5001/completion", json=params, headers={"Content-Type": "application/json"})
            response = query.json()

            st.markdown(response["completion"])
            st.text(f"Generation done in {response['compute_time']:.3} s.")


if __name__ == "__main__":
    main()
