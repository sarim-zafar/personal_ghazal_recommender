import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import PowerTransformer
import pyzipper
import warnings

warnings.filterwarnings("ignore")

# --- [Unchanged: decrypt_data, load_data] ---
def decrypt_data():
    # Decrypt the ZIP file with the password
    # Ensure 'data/data.zip' path is correct and secrets are configured
    try:
        with open('data/data.zip', 'rb') as f_in:
            with pyzipper.AESZipFile(f_in) as f_zip:
                password = st.secrets.my_cool_secrets.get('pwd')
                if not password:
                    st.error("Password not found in Streamlit secrets. Please check your secrets configuration.")
                    st.stop()
                f_zip.setpassword(bytes(password, 'UTF-8'))
                # Extract directly to memory or a temporary location if needed,
                # writing directly to the root might cause issues depending on deployment.
                # For simplicity, keeping original extraction logic, but be mindful of write permissions.
                for name in f_zip.namelist():
                    data = f_zip.read(name)
                    with open(name, 'wb') as f_out:
                        f_out.write(data)
        return 1
    except FileNotFoundError:
        st.error("Error: 'data/data.zip' not found. Please ensure the file exists.")
        st.stop()
    except pyzipper.zipfile.BadZipFile:
        st.error("Error: Failed to open ZIP file. It might be corrupted or not a ZIP file.")
        st.stop()
    except RuntimeError as e:
         if "Bad password" in str(e):
             st.error("Error: Incorrect password provided for decrypting the data file.")
             st.stop()
         else:
             st.error(f"An unexpected error occurred during decryption: {e}")
             st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()


@st.cache_data
def load_data():
    # Decryption should ideally happen once if needed, maybe outside cache or managed carefully
    # decrypt_data() # Uncomment if decryption is needed every time data is loaded (usually not ideal)
    try:
        # Make sure 'data.parquet' exists after potential decryption
        df = pd.read_parquet("data.parquet")
        emb = np.array(df['embedding'].to_list())
        # Handle potential errors during transformation
        if emb.ndim == 1: # Reshape if it's somehow 1D
             emb = emb.reshape(-1, 1)
        elif emb.shape[1] == 0: # Check for empty embeddings
             st.error("Error: Embeddings data is empty.")
             st.stop()

        pt = PowerTransformer()
        emb = pt.fit_transform(emb)
        emb = emb.astype(np.float32)
        return df, emb
    except FileNotFoundError:
        st.error("Error: 'data.parquet' not found. Decryption might have failed or the file is missing.")
        # Attempt decryption here if file not found initially
        st.warning("Attempting to decrypt data file...")
        if decrypt_data():
            try:
                df = pd.read_parquet("data.parquet")
                emb = np.array(df['embedding'].to_list())
                if emb.ndim == 1: emb = emb.reshape(-1, 1)
                if emb.shape[1] == 0:
                    st.error("Error: Embeddings data is empty after decryption.")
                    st.stop()
                pt = PowerTransformer()
                emb = pt.fit_transform(emb)
                emb = emb.astype(np.float32)
                st.success("Data decrypted and loaded successfully.")
                return df, emb
            except Exception as e:
                st.error(f"Error loading data after decryption: {e}")
                st.stop()
        else:
            st.error("Decryption failed. Cannot load data.")
            st.stop()

    except Exception as e:
        st.error(f"An error occurred during data loading or processing: {e}")
        st.stop()

# --- [Unchanged: set_page_config, set_custom_css] ---
def set_page_config():
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
            line-height: 2.5;
        }

        /* Specific style for English text */
        .eng_text {
            font-family: 'Roboto', sans-serif !important;
            text-align: left;
            line-height: 2.5;
        }

        /* Ensure markdown headers using the class get the font */
        h1.urdu_text, h2.urdu_text, h3.urdu_text, h4.urdu_text, h5.urdu_text, h6.urdu_text {
             font-family: 'Noto Nastaliq Urdu', serif !important;
        }
         h1.eng_text, h2.eng_text, h3.eng_text, h4.eng_text, h5.eng_text, h6.eng_text {
             font-family: 'Roboto', sans-serif !important;
        }

        /* Target the button element itself within Streamlit's structure */
        /* This applies the font to ALL text inside the button */
        div[data-testid="stButton"] > button {
            font-family: 'Noto Nastaliq Urdu', 'Roboto', sans-serif !important; /* Prioritize Nastaliq, fallback */
             /* Adjust padding or other properties if needed for Nastaliq */
             padding-top: 0.25rem;
             padding-bottom: 0.35rem; /* Slight adjustment might be needed */
        }

        /* Style for the author name below the progress bar */
        .author-meta {
            margin-top: 0.1rem; /* Adjust spacing above author name */
            margin-bottom: 0.5rem; /* Adjust spacing below author name */
        }

        /* Style for the AI certainty label */
        .certainty-label {
            margin-bottom: 0.1rem; /* Reduce space between label and progress bar */
            text-align: right; /* Ensure right alignment */
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

# --- [Unchanged: display_buttons] ---
def display_buttons(buttons_placeholder, sep):
    sep.write("---")

    options = {
        "Thumbs Up": "👍",
        "Skip": "⏩",
        "Thumbs Down": "👎",
    }
    selected_option = None
    # Adjust columns if needed, or keep as is
    cols = buttons_placeholder.columns([1, 1, 1, 1.5, 1.5, 1.5, 1, 1, 1])

    # Labels are plain strings. The CSS rule handles the font.
    bt1 = cols[3].button(f'{options["Thumbs Up"]} پسند', use_container_width=True, key="btn_up")
    bt2 = cols[4].button(f'{options["Skip"]} چھوڑیں', use_container_width=True, key="btn_skip")
    bt3 = cols[5].button(f'{options["Thumbs Down"]} ناپسند', use_container_width=True, key="btn_down")

    if bt1:
        selected_option = "Thumbs Up"
    elif bt2:
        selected_option = "Skip"
    elif bt3:
        selected_option = "Thumbs Down"
    return selected_option

# --- [Unchanged: update_choices, display_recommendations (except parameters)] ---
def update_choices(selected_option):
    if 'idx' in st.session_state and st.session_state.idx is not None:
        st.session_state.choices[selected_option].append(st.session_state.idx)
    else:
        st.warning("No Ghazal index found in session state to record choice against.")

# Modified: Removed author_placeholder parameter
def display_recommendations(title_placeholder, sep_1_placeholder, text_placeholder):
    # Ensure session state variables exist before accessing
    title = st.session_state.get('title', 'Title Not Found')
    # Author is now displayed separately in main()
    ghazal_eng = st.session_state.get('ghazal_eng', 'English text not available.')
    ghazal = st.session_state.get('ghazal', 'Urdu text not available.')

    title_placeholder.markdown(f'<h1 class="urdu_text">{title}</h1>',
                                 unsafe_allow_html=True)
    # Author display removed from here
    sep_1_placeholder.write('---')
    txt_col1, txt_col2 = text_placeholder.columns(2)

    # Apply classes for specific font styling
    txt_col1.markdown(f'<div class="eng_text">{ghazal_eng}</div>',
                                 unsafe_allow_html=True)
    txt_col2.markdown(f'<div class="urdu_text">{ghazal}</div>',
                                 unsafe_allow_html=True)


# --- [Unchanged: init_choice_tracker, filter_choices] ---
def init_choice_tracker():
    if 'choices' not in st.session_state:
        st.session_state.choices = {
            "Thumbs Up": [],
            "Skip": [],
            "Thumbs Down": []
        }
    # Initialize other session state variables used
    if 'idx' not in st.session_state:
        st.session_state.idx = None
    if 'proba' not in st.session_state:
        st.session_state.proba = 0.0 # Use float for probability
    if 'title' not in st.session_state:
        st.session_state.title = ""
    if 'author' not in st.session_state:
        st.session_state.author = ""
    if 'ghazal' not in st.session_state:
        st.session_state.ghazal = ""
    if 'ghazal_eng' not in st.session_state:
        st.session_state.ghazal_eng = ""
    # Added missing initialization from previous step
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None

def filter_choices(choices):
    # Ensure choices is a mutable list
    choices_list = list(choices)
    seen = set(
        st.session_state.choices["Thumbs Down"] +
        st.session_state.choices["Skip"] +
        st.session_state.choices["Thumbs Up"]
    )
    filtered_choices = [idx for idx in choices_list if idx not in seen]
    return filtered_choices

# --- [Unchanged: ML Training/Prediction functions] ---
# Placeholder for global variable 'emb', ensure it's assigned in load_data
emb = None

@st.cache_data
def train_one(_X_shape, i, C): # Pass shape or hash of X instead of X itself if X is large
    global emb # Relies on global emb variable being loaded
    if emb is None:
        st.error("Embeddings not loaded for training.")
        return None

    y = np.zeros(emb.shape[0])
    y[i] = 1
    clf = svm.LinearSVC(class_weight='balanced',
                    verbose=False,
                    max_iter=10_000, tol=1e-6, C=C, random_state=8)
    clf.fit(emb, y)
    return clf

def train_exemplar_svm(X, pos_examples, C=0.25):
    exemplar_svms = []
    if not pos_examples:
        return []
    for i in pos_examples:
        clf = train_one(X.shape, i, C)
        if clf:
             exemplar_svms.append(clf)
    return exemplar_svms

def normalize_decision_values(decision_values):
    max_val = np.max(decision_values)
    if np.isneginf(max_val):
         return np.zeros_like(decision_values)
    if np.isinf(max_val):
        probs = np.zeros_like(decision_values)
        inf_indices = np.isinf(decision_values)
        if np.sum(inf_indices) > 0: # Avoid division by zero if no infs (shouldn't happen here)
             probs[inf_indices] = 1.0 / np.sum(inf_indices)
        return probs

    norm_values = decision_values - max_val
    # Handle potential overflow in exp
    norm_values = np.clip(norm_values, -700, 700) # Clip values to prevent overflow
    exp_values = np.exp(norm_values)
    sum_exp_values = np.sum(exp_values)

    if sum_exp_values == 0 or np.isnan(sum_exp_values) or np.isinf(sum_exp_values):
        return np.zeros_like(decision_values)

    return exp_values / sum_exp_values

def get_most_conf(top_preds):
    if not top_preds:
        return None, 0.0
    valid_preds = [p for p in top_preds if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[1], (int, float)) and not np.isnan(p[1])]
    if not valid_preds:
        return None, 0.0
    try:
        max_idx, max_score = max(valid_preds, key=lambda x: x[1])
    except ValueError:
        return None, 0.0
    return max_idx, max_score * 100

def get_prediction(_exemplar_svms, X, all_indices):
    if not _exemplar_svms:
        return None, 0.0

    top_preds = []
    for svm_idx, svm in enumerate(_exemplar_svms):
        try:
            decision_vals = svm.decision_function(X)
            if np.any(np.isnan(decision_vals)) or np.any(np.isinf(decision_vals)):
                continue
            certainty_scores = normalize_decision_values(decision_vals)
        except Exception as e:
            continue

        indexed_scores = list(zip(all_indices, certainty_scores))
        sorted_indexed_scores = sorted(indexed_scores, key=lambda item: item[1], reverse=True)
        available_choices = filter_choices([idx for idx, score in sorted_indexed_scores])

        best_match = None
        best_match_confidence = 0.0
        for idx, score in sorted_indexed_scores:
            if idx in available_choices:
                if isinstance(score, (int, float)) and not np.isnan(score):
                     best_match = idx
                     best_match_confidence = score
                     break

        if best_match is not None:
            top_preds.append([best_match, best_match_confidence])

    top_pred_idx, conf = get_most_conf(top_preds)
    return top_pred_idx, conf

def get_recommendation(emb, all_indices):
    pos_examples = st.session_state.choices.get("Thumbs Up", [])
    if not pos_examples:
        return 0.0, None

    clfs = train_exemplar_svm(emb, pos_examples)
    if not clfs:
        return 0.0, None

    predicted_idx, certainty_score = get_prediction(clfs, emb, all_indices)
    if not isinstance(certainty_score, (int, float)):
        certainty_score = 0.0
    return certainty_score, predicted_idx

# --- [Modified: get_ghazal] ---
# Modified: Removed author_placeholder from parameters
def get_ghazal(df, emb, title_placeholder, sep_1_placeholder, text_placeholder):

    all_indices = df.index.values.tolist()
    available_indices = filter_choices(all_indices)

    if not available_indices:
        st.warning("No more Ghazals to recommend based on your feedback!")
        # Clear placeholders
        title_placeholder.empty()
        sep_1_placeholder.empty()
        text_placeholder.empty()
        # We don't clear meta_placeholder here, main() will do it
        st.session_state.idx = None # Reset index
        st.session_state.proba = 0.0 # Reset probability
        st.session_state.author = "" # Clear author too
        return 0 # Indicate failure or completion

    current_rec_idx = None
    current_rec_proba = 0.0

    # Cold start: If no positive feedback yet, choose randomly from available
    if not st.session_state.choices.get("Thumbs Up"):
        current_rec_idx = np.random.choice(available_indices)
        current_rec_proba = 0.0 # No confidence score for random choice
    else:
        # Get ML-based recommendation
        proba, rec_idx = get_recommendation(emb, all_indices)
        if rec_idx is not None and rec_idx in available_indices:
            current_rec_idx = rec_idx
            current_rec_proba = max(0.0, min(100.0, proba if isinstance(proba, (int, float)) else 0.0))
        else:
            # Fallback if ML fails or recommends an unavailable item
            current_rec_idx = np.random.choice(available_indices)
            current_rec_proba = 0.0 # Reset probability for random choice

    # Update session state with the determined index and probability
    st.session_state.idx = current_rec_idx
    st.session_state.proba = current_rec_proba / 100.0 # Store as 0-1 probability

    # Fetch ghazal data and store in session state
    if st.session_state.idx is not None:
        try:
            ghazal_data = df.loc[st.session_state.idx] # Use .loc for index lookup
            st.session_state.title = ghazal_data.title.strip()
            st.session_state.author = ghazal_data.author.strip() # Store author
            st.session_state.ghazal = ghazal_data.text.replace('\n', '<br>').strip()
            st.session_state.ghazal_eng = ghazal_data.eng_text.replace('\n', '<br>').strip()

            # Display the non-author parts
            display_recommendations(title_placeholder, sep_1_placeholder, text_placeholder)
            return 1 # Indicate success
        except KeyError:
             st.error(f"Error: Index {st.session_state.idx} not found in DataFrame.")
             st.session_state.idx = None # Reset index if invalid
             st.session_state.proba = 0.0
             st.session_state.author = ""
             return 0 # Indicate failure
        except Exception as e:
            st.error(f"An unexpected error occurred while fetching Ghazal data for index {st.session_state.idx}: {e}")
            st.session_state.idx = None
            st.session_state.proba = 0.0
            st.session_state.author = ""
            return 0
    else:
        st.error("Failed to select a valid Ghazal index.")
        st.session_state.proba = 0.0
        st.session_state.author = ""
        return 0

# --- [Modified: setup_carcass] ---
def setup_carcass():
    # Placeholders for dynamic content
    title_placeholder = st.empty()
    # NEW: Placeholder for meta information (Score Label, Progress, Author)
    meta_placeholder = st.empty()
    sep_1_placeholder = st.empty()
    text_placeholder = st.empty()
    sep_2_placeholder = st.empty()
    buttons_placeholder = st.empty()

    # Return placeholders including the new meta_placeholder
    return [title_placeholder, meta_placeholder, sep_1_placeholder,
            text_placeholder, sep_2_placeholder, buttons_placeholder]


# --- [Modified: main function] ---
def main():
    # Setup app configuration and CSS
    set_page_config()
    set_custom_css() # Load fonts and styles

    # Initialize session state variables
    init_choice_tracker()

    # Load data (cached) - Assign to global 'emb' as well for train_one
    global emb
    df, emb_loaded = load_data()
    if df is None or emb_loaded is None:
        st.error("Failed to load data. Application cannot proceed.")
        st.stop() # Stop execution if data loading failed
    emb = emb_loaded # Assign loaded embeddings to the global variable

    # Setup UI layout placeholders
    (title_placeholder, meta_placeholder, sep_1_placeholder,
     text_placeholder, sep_2_placeholder, buttons_placeholder) = setup_carcass()

    # --- Process Previous Selection ---
    if st.session_state.selected_option:
        update_choices(st.session_state.selected_option)
        st.session_state.selected_option = None # Reset after processing

    # --- Get Current Ghazal Data ---
    get_ghazal_success = get_ghazal(df, emb, title_placeholder,
                                    sep_1_placeholder, text_placeholder)

    # --- Display UI Elements (Score, Author, Buttons) ---
    selected_option_this_run = None # Reset for this run
    if get_ghazal_success:
         # Use the meta_placeholder to display score label, progress bar, and author
         with meta_placeholder.container():
             # 1. Display Author Name (Full width relative to container, below the progress bar row)
             author_name = st.session_state.get("author", "")
             if author_name: # Only display if author exists
                 st.markdown(f'<h5 class="urdu_text author-meta">{author_name}</h5>', unsafe_allow_html=True)

             # 2. Display AI Certainty Label (Full width relative to container)
             st.markdown('<p class="urdu_text certainty-label;">مصنوعی ذہانت کی یہ تجویز کی یقینیت</p>', unsafe_allow_html=True)

             # 3. Create columns ONLY for the progress bar
             # Use [1, 1, 1] for three equal columns (1/3 width each)
             # Or adjust ratios like [1, 2] if you want 1/3 on right, 2/3 empty space left
             # Using [1, 1, 1] is simpler for exactly 1/3
             progress_cols = st.columns([1, 1, 1])

             # Place the progress bar in the first column (which is the rightmost one due to RTL)
             with progress_cols[2]:
                 st.progress(st.session_state.get('proba', 0.0))
             # progress_cols[1] and progress_cols[2] will be empty, creating space

             

         # Display feedback buttons and capture if one was clicked *in this run*
         selected_option_this_run = display_buttons(buttons_placeholder, sep_2_placeholder)
    else:
        # If no ghazal could be displayed, clear the meta area and buttons
        meta_placeholder.empty()
        buttons_placeholder.empty()
        sep_2_placeholder.empty()

    # --- Handle Button Click for the *Next* Rerun ---
    if selected_option_this_run:
        st.session_state.selected_option = selected_option_this_run
        st.rerun() # Trigger rerun to process choice and show next recommendation

if __name__ == "__main__":
    main()