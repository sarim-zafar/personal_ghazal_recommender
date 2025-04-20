# ui_components.py
import streamlit as st

def set_page_config():
    """Sets the Streamlit page configuration."""
    st.set_page_config(
        page_title="شخصیات بندی شدہ غزل کی تجویز کنندہ",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'About': """
                    Every time you open the app it will start by giving
                    recommendation’s randomly but if you give it feedback at the bottom
                    the next recommendation will be made using that information.
                    And it will retain this information until you hit refresh or close the
                    tab and open it again
            """
        }
    )

def set_custom_css():
    """Applies custom CSS for fonts and layout adjustments."""
    # --- Load Fonts from Google Fonts ---
    nastaliq_font_url = 'https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400..700&display=swap'
    roboto_font_url = "https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap"

    st.markdown(f'<link href="{nastaliq_font_url}" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown(f'<link href="{roboto_font_url}" rel="stylesheet">', unsafe_allow_html=True)

    # --- Custom CSS for Fonts ---
    st.markdown(
        """
        <style>
        /* Base font is set in config.toml - Roboto */

        /* Specific style for Urdu text */
        .urdu_text {
            font-family: 'Noto Nastaliq Urdu', serif !important;
            direction: rtl;
            text-align: right;
            line-height: 2.25 !important; /* Increased line height for Urdu */
        }

        /* Specific style for English text */
        .eng_text {
            font-family: 'Roboto', sans-serif !important;
            text-align: left;
            line-height: 2.0 !important; /* Adjusted line height for English */
        }

        /* Ensure markdown headers using the class get the font */
        h1.urdu_text, h2.urdu_text, h3.urdu_text, h4.urdu_text, h5.urdu_text, h6.urdu_text {
             font-family: 'Noto Nastaliq Urdu', serif !important;
        }
         h1.eng_text, h2.eng_text, h3.eng_text, h4.eng_text, h5.eng_text, h6.eng_text {
             font-family: 'Roboto', sans-serif !important;
        }

        /* Style for the author name below the progress bar */
        .author-meta {
            margin-top: 0.1rem; /* Adjust spacing above author name */
            margin-bottom: 0.5rem; /* Adjust spacing below author name */
            font-size: 1.1em; /* Slightly larger author font */
        }

        /* Style for the AI certainty label */
        .certainty-label {
            margin-bottom: 0.1rem; /* Reduce space between label and progress bar */
            text-align: right; /* Ensure right alignment */
            font-size: 0.9em; /* Slightly smaller certainty label */
            color: #555; /* Dimmed color for the label */
        }

        /* Style for the ghazal text for better readability */
        h5.urdu_text, h5.eng_text {
             font-size: 1.15em !important; /* Slightly larger ghazal text */
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

def display_buttons(buttons_placeholder, sep):
    """Displays the feedback buttons (Like, Skip, Dislike)."""
    sep.write("---") # Separator above buttons

    options = {
        "Thumbs Up": "👍",
        "Skip": "⏩",
        "Thumbs Down": "👎",
    }
    selected_option = None
    # Columns for button layout: [spacer, spacer, spacer, button, button, button, spacer, spacer, spacer]
    cols = buttons_placeholder.columns([1, 1, 1, 1.5, 1.5, 1.5, 1, 1, 1])

    # Labels are plain strings. The CSS rule handles the font.
    # Using CSS classes for potential future styling if needed
    bt1 = cols[3].button(f'{options["Thumbs Up"]} پسند', use_container_width=True, key="btn_up", help="Mark this Ghazal as liked")
    bt2 = cols[4].button(f'{options["Skip"]} چھوڑیں', use_container_width=True, key="btn_skip", help="Skip this Ghazal and show another")
    bt3 = cols[5].button(f'{options["Thumbs Down"]} ناپسند', use_container_width=True, key="btn_down", help="Mark this Ghazal as disliked")

    if bt1:
        selected_option = "Thumbs Up"
    elif bt2:
        selected_option = "Skip"
    elif bt3:
        selected_option = "Thumbs Down"

    return selected_option


def display_recommendations(title_placeholder, sep_1_placeholder, text_placeholder):
    """Displays the Ghazal title, text (Urdu and English)."""
    # Ensure session state variables exist before accessing
    title = st.session_state.get('title', 'عنوان نہیں ملا') # "Title Not Found"
    ghazal_eng = st.session_state.get('ghazal_eng', 'English text not available.')
    ghazal = st.session_state.get('ghazal', 'اردو متن دستیاب نہیں ہے۔') # "Urdu text not available."

    title_placeholder.markdown(f'<h1 class="urdu_text" style="text-align: center;">{title}</h1>',
                                 unsafe_allow_html=True) # Centered title
    sep_1_placeholder.write('---') # Separator below title
    txt_col1, txt_col2 = text_placeholder.columns(2)

    # Apply classes for specific font styling and enhanced readability
    txt_col1.markdown(f'<h5 class="eng_text">{ghazal_eng}</h5>',
                                 unsafe_allow_html=True)
    txt_col2.markdown(f'<h5 class="urdu_text">{ghazal}</h5>',
                                 unsafe_allow_html=True)


def setup_carcass():
    """Sets up the main layout placeholders for dynamic content."""
    title_placeholder = st.empty()
    meta_placeholder = st.empty() # For author, score label, progress bar
    sep_1_placeholder = st.empty() # Separator below title
    text_placeholder = st.empty() # For Ghazal text columns
    sep_2_placeholder = st.empty() # Separator above buttons
    buttons_placeholder = st.empty() # For feedback buttons

    return [title_placeholder, meta_placeholder, sep_1_placeholder,
            text_placeholder, sep_2_placeholder, buttons_placeholder]