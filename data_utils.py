# data_utils.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pyzipper
import warnings

warnings.filterwarnings("ignore")

def decrypt_data():
    """Decrypts the data.zip file using the password from secrets."""
    try:
        with open('data/data.zip', 'rb') as f_in:
            with pyzipper.AESZipFile(f_in) as f_zip:
                password = st.secrets.my_cool_secrets.get('pwd')
                if not password:
                    st.error("Password not found in Streamlit secrets. Please check your secrets configuration.")
                    st.stop()
                f_zip.setpassword(bytes(password, 'UTF-8'))
                for name in f_zip.namelist():
                    data = f_zip.read(name)
                    # Extract to the root directory where the app runs
                    with open(name, 'wb') as f_out:
                        f_out.write(data)
        return True # Indicate success
    except FileNotFoundError:
        st.error("Error: 'data/data.zip' not found. Please ensure the file exists in the 'data' directory relative to the app.")
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
        st.error(f"An unexpected error occurred during decryption: {e}")
        st.stop()
    return False # Indicate failure


@st.cache_data
def load_data():
    """Loads and preprocesses the data from data.parquet, attempting decryption if necessary."""
    parquet_file = "data.parquet"
    try:
        # Try loading directly first
        df = pd.read_parquet(parquet_file)
    except FileNotFoundError:
        st.warning(f"'{parquet_file}' not found. Attempting to decrypt 'data/data.zip'...")
        if decrypt_data():
            st.success("Data decrypted successfully. Loading data...")
            try:
                df = pd.read_parquet(parquet_file)
            except FileNotFoundError:
                st.error(f"Error: '{parquet_file}' still not found after attempting decryption.")
                st.stop()
            except Exception as e:
                st.error(f"Error loading '{parquet_file}' after decryption: {e}")
                st.stop()
        else:
            st.error("Decryption failed. Cannot load data.")
            st.stop() # Stop if decryption failed
    except Exception as e:
        st.error(f"An error occurred during initial data loading: {e}")
        st.stop()

    # --- Process Data ---
    try:
        emb = np.array(df['embedding'].to_list())

        # Handle potential errors during transformation
        if emb.ndim == 1: # Reshape if it's somehow 1D
             emb = emb.reshape(-1, 1)
        if emb.size == 0: # Check for empty embeddings array
             st.error("Error: Embeddings data is empty or could not be loaded correctly.")
             st.stop()
        if emb.shape[1] == 0: # Check for zero columns after reshape
             st.error("Error: Embeddings data has zero features.")
             st.stop()

        # print(np.max(emb),np.min(emb),np.mean(emb))
        pt = MinMaxScaler()
        # Ensure there's data to fit
        if emb.shape[0] > 0:
            emb = pt.fit_transform(emb)
        else:
            st.warning("No data to fit PowerTransformer.")
            # Keep emb as potentially empty array, handle downstream if needed

        # Ensure float32 for efficiency
        emb = emb.astype(np.float32)
        # print(np.max(emb),np.min(emb),np.mean(emb))
        return df, emb

    except KeyError as e:
        st.error(f"Error accessing column in DataFrame: {e}. Is the '{parquet_file}' file correct?")
        st.stop()
    except ValueError as e:
        st.error(f"Error during data transformation (e.g., PowerTransformer): {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during data processing: {e}")
        st.stop()