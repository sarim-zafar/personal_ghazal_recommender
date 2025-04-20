# app.py
import streamlit as st
import numpy as np
import pandas as pd # Keep pandas import if df is used directly here

# Import functions from custom modules
from data_utils import load_data
from ui_components import (set_page_config, set_custom_css, setup_carcass,
                           display_recommendations, display_buttons)
from state_utils import init_choice_tracker, update_choices, filter_choices
from recommendation import get_recommendation

def get_ghazal(df, emb, title_placeholder, sep_1_placeholder, text_placeholder):
    """
    Selects the next Ghazal to display, either randomly or based on recommendation.
    Updates session state with the selected Ghazal's details.

    Args:
        df (pd.DataFrame): The DataFrame containing Ghazal data.
        emb (np.ndarray): The embeddings matrix.
        title_placeholder: Streamlit placeholder for the title.
        sep_1_placeholder: Streamlit placeholder for the first separator.
        text_placeholder: Streamlit placeholder for the text columns.

    Returns:
        int: 1 if a Ghazal was successfully selected and displayed, 0 otherwise.
    """
    all_indices = df.index.tolist() # Use tolist() for a standard list
    available_indices = filter_choices(all_indices) # Get indices not yet seen

    if not available_indices:
        st.warning("تمام غزلیں دیکھ لی گئیں۔ آپ کی آراء کی بنیاد پر مزید سفارشات نہیں ہیں!")
        # Clear placeholders if nothing to show
        title_placeholder.empty()
        sep_1_placeholder.empty()
        text_placeholder.empty()
        st.session_state.idx = None
        st.session_state.proba = 0.0
        st.session_state.author = ""
        return 0 # Indicate completion

    current_rec_idx = None
    current_rec_proba = 0.0 # Probability score (0-100)

    # Check if *any* feedback (like or dislike) has been given
    has_feedback = bool(st.session_state.choices.get("Thumbs Up")) or \
                   bool(st.session_state.choices.get("Thumbs Down"))

    if not has_feedback:
        # Cold start: No feedback yet, choose randomly from available
        # st.info("ابھی تک کوئی رائے موصول نہیں ہوئی، ایک بے ترتیب غزل دکھائی جا رہی ہے۔") # Optional: message for random choice
        current_rec_idx = np.random.choice(available_indices)
        current_rec_proba = 0.0 # No confidence score for random choice
    else:
        # Get ML-based recommendation using Label Spreading
        proba, rec_idx = get_recommendation(emb, all_indices) # proba is 0-100

        if rec_idx is not None and rec_idx in available_indices:
            # Valid recommendation found and it's available
            current_rec_idx = rec_idx
            current_rec_proba = max(0.0, min(100.0, proba if isinstance(proba, (int, float)) and not np.isnan(proba) else 0.0))
        else:
            # Fallback: ML failed, recommended seen item, or only dislikes given etc.
            # Choose randomly from the *remaining available* options
            # st.info("ماڈل کوئی نئی تجویز نہیں دے سکا یا تجویز کردہ غزل پہلے ہی دیکھی جا چکی ہے، ایک بے ترتیب غزل دکھائی جا رہی ہے۔") # Optional message
            if available_indices: # Ensure there are still items to choose from
                current_rec_idx = np.random.choice(available_indices)
            else:
                 # This case should technically be caught earlier, but handle defensively
                 st.warning("Recommendation failed and no available items remain for random choice.")
                 return 0 # Cannot select an item
            current_rec_proba = 0.0 # Reset probability for random choice

    # --- Update session state with the selected Ghazal ---
    st.session_state.idx = current_rec_idx
    # Store probability as 0-1 for st.progress
    st.session_state.proba = current_rec_proba / 100.0

    # Fetch Ghazal data from DataFrame and store in session state
    if st.session_state.idx is not None:
        try:
            # Use .loc for safe index-based lookup
            ghazal_data = df.loc[st.session_state.idx]
            st.session_state.title = ghazal_data.get('title', 'No Title').strip()
            st.session_state.author = ghazal_data.get('author', 'Unknown Author').strip()
            st.session_state.ghazal = ghazal_data.get('text', 'No Urdu Text').replace('\n', '<br>').strip()
            st.session_state.ghazal_eng = ghazal_data.get('eng_text', 'No English Text').replace('\n', '<br>').strip()

            # Display the fetched Ghazal (Title, Separator, Text)
            display_recommendations(title_placeholder, sep_1_placeholder, text_placeholder)
            return 1 # Indicate success

        except KeyError:
             st.error(f"Error: Index {st.session_state.idx} not found in DataFrame. Resetting.")
             st.session_state.idx = None # Reset invalid index
             st.session_state.proba = 0.0
             st.session_state.author = ""
             return 0 # Indicate failure
        except AttributeError as ae:
             st.error(f"Error accessing data for index {st.session_state.idx}: {ae}. Check DataFrame structure.")
             st.session_state.idx = None
             st.session_state.proba = 0.0
             st.session_state.author = ""
             return 0
        except Exception as e:
            st.error(f"An unexpected error occurred while fetching/displaying Ghazal data for index {st.session_state.idx}: {e}")
            st.session_state.idx = None
            st.session_state.proba = 0.0
            st.session_state.author = ""
            return 0
    else:
        # This case might occur if random choice failed or rec_idx became None unexpectedly
        st.error("Failed to select a valid Ghazal index.")
        st.session_state.proba = 0.0
        st.session_state.author = ""
        return 0


def main():
    """Main function to run the Streamlit application."""
    # --- Initial Setup ---
    set_page_config()
    set_custom_css()
    init_choice_tracker() # Ensure session state exists

    # --- Load Data ---
    # load_data is cached, so it runs efficiently on reruns
    # It handles decryption if needed.
    df, emb = load_data()
    if df is None or emb is None or emb.size == 0:
        # Errors are handled within load_data, but double-check
        st.error("Application cannot proceed without valid data and embeddings.")
        st.stop()

    # --- Setup UI Placeholders ---
    (title_placeholder, meta_placeholder, sep_1_placeholder,
     text_placeholder, sep_2_placeholder, buttons_placeholder) = setup_carcass()

    # --- Process Feedback from *Previous* Rerun ---
    # Check if a button was clicked in the previous interaction cycle
    if 'selected_option' in st.session_state and st.session_state.selected_option:
        update_choices(st.session_state.selected_option)
        # Reset the flag *after* processing so it doesn't trigger again unintentionally
        st.session_state.selected_option = None

    # --- Get and Display Current Ghazal ---
    # This function now contains the logic to choose randomly or via recommendation
    get_ghazal_success = get_ghazal(df, emb, title_placeholder,
                                    sep_1_placeholder, text_placeholder)

    # --- Display Metadata and Buttons ---
    selected_option_this_run = None # To capture button click *in this specific run*
    if get_ghazal_success:
        # Display Author and Confidence Score (if applicable)
        with meta_placeholder.container():
            author_name = st.session_state.get("author", "")
            if author_name:
                 st.markdown(f'<h5 class="urdu_text author-meta" style="text-align: center;">{author_name}</h5>', unsafe_allow_html=True) # Centered Author

            # Only show certainty if it's > 0 (i.e., not a purely random choice)
            current_proba = st.session_state.get('proba', 0.0) # Proba is 0-1
            if current_proba > 0.0:
                st.markdown('<p class="urdu_text certainty-label;">تجویز کی یقینیت</p>', unsafe_allow_html=True) # "Recommendation Certainty"
                # Use columns to control progress bar width (place in rightmost column)
                progress_cols = st.columns([1, 1, 1]) # [space, space, bar]
                with progress_cols[2]:
                    st.progress(current_proba)
            else:
                 # Add a little space if no progress bar is shown
                 st.markdown("<br>", unsafe_allow_html=True)

        # Display feedback buttons and capture the choice made *in this run*
        selected_option_this_run = display_buttons(buttons_placeholder, sep_2_placeholder)
    else:
        # Clear meta/button areas if no Ghazal could be displayed (e.g., all seen)
        meta_placeholder.empty()
        buttons_placeholder.empty()
        sep_2_placeholder.empty()

    # --- Handle Button Click for the *Next* Rerun ---
    if selected_option_this_run:
        # Store the button clicked *in this run* into session state.
        # It will be processed by update_choices() at the *start* of the next rerun.
        st.session_state.selected_option = selected_option_this_run
        # Trigger the rerun to process the choice and get the next recommendation.
        st.rerun()

if __name__ == "__main__":
    main()