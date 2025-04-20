# state_utils.py
import streamlit as st

def init_choice_tracker():
    """Initializes session state variables if they don't exist."""
    if 'choices' not in st.session_state:
        st.session_state.choices = {
            "Thumbs Up": [],
            "Skip": [],
            "Thumbs Down": []
        }
    # Initialize other session state variables used
    defaults = {
        'idx': None,
        'proba': 0.0, # Use float for probability (0-1 range)
        'title': "",
        'author': "",
        'ghazal': "",
        'ghazal_eng': "",
        'selected_option': None # Tracks button click between reruns
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_choices(selected_option):
    """Updates the choices lists in session state based on user feedback."""
    if 'idx' in st.session_state and st.session_state.idx is not None:
        current_idx = st.session_state.idx
        choices_dict = st.session_state.choices

        # Prevent adding the same index multiple times to the *same* list
        if current_idx not in choices_dict[selected_option]:
            choices_dict[selected_option].append(current_idx)

        # Ensure an item isn't in multiple lists (like/dislike overrides skip)
        if selected_option == "Thumbs Up":
            if current_idx in choices_dict["Thumbs Down"]:
                choices_dict["Thumbs Down"].remove(current_idx)
            if current_idx in choices_dict["Skip"]:
                choices_dict["Skip"].remove(current_idx)
        elif selected_option == "Thumbs Down":
            if current_idx in choices_dict["Thumbs Up"]:
                choices_dict["Thumbs Up"].remove(current_idx)
            if current_idx in choices_dict["Skip"]:
                choices_dict["Skip"].remove(current_idx)
        elif selected_option == "Skip":
             # If skipped, it shouldn't be in like/dislike already, but check just in case
            if current_idx in choices_dict["Thumbs Up"]:
                 choices_dict["Thumbs Up"].remove(current_idx)
            if current_idx in choices_dict["Thumbs Down"]:
                 choices_dict["Thumbs Down"].remove(current_idx)

    else:
        # This might happen if the page refreshes unexpectedly before idx is set
        st.warning("Cannot record choice: Ghazal index not found in session state.")

def filter_choices(all_indices):
    """Filters out indices that have already been interacted with (liked, disliked, or skipped)."""
    # Ensure all_indices is a list or set for efficient processing
    if not isinstance(all_indices, (list, set)):
         all_indices = list(all_indices)

    seen = set(
        st.session_state.choices.get("Thumbs Down", []) +
        st.session_state.choices.get("Skip", []) +
        st.session_state.choices.get("Thumbs Up", [])
    )
    # Return a list of indices that are not in the 'seen' set
    filtered_choices = [idx for idx in all_indices if idx not in seen]
    return filtered_choices