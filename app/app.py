import os
import streamlit as st
import streamlit.components.v1 as components

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="NASA SMAP Spacecraft Health & Telemetry Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ingest style overrides to hide Streamlit header, footer, padding, and margins
st.markdown("""
    <style>
        /* Hide Streamlit elements */
        header {visibility: hidden; height: 0px !important;}
        footer {visibility: hidden; height: 0px !important;}
        #MainMenu {visibility: hidden;}
        div.block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }
        iframe {
            border: none;
            width: 100vw;
            height: 100vh;
        }
    </style>
""", unsafe_allow_html=True)

# Set path reference
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# File path definitions
html_path = os.path.join(ROOT_DIR, "index.html")
json_path = os.path.join(ROOT_DIR, "scenario_data.json")

def main():
    if not os.path.exists(html_path):
        st.error(f"Missing core HTML asset: {html_path}")
        return
        
    if not os.path.exists(json_path):
        st.error(f"Missing precomputed scenario data: {json_path}")
        return

    # Load Template
    with open(html_path, "r", encoding="utf-8") as f:
        html_template = f.read()

    # Load JSON dataset
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = f.read()

    # Inject dynamic scenario variables
    html_content = html_template.replace("{{SCENARIO_DATA_JSON}}", json_data)
    html_content = html_content.replace("{{UPLOADED_SCENARIO_NAME}}", "null")

    # Render Dashboard inside Streamlit (using full-screen viewport dimensions)
    # The height is configured to 980px to fully show charts and logs without double scrolls.
    components.html(html_content, height=980, scrolling=True)

if __name__ == "__main__":
    main()
