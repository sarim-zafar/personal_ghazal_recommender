# recommendation.py
import streamlit as st
import numpy as np
from sklearn.semi_supervised import LabelSpreading

# Import filter_choices from state_utils (assuming it's in the same directory)
from state_utils import filter_choices

def get_recommendation(emb, all_indices):
    """
    Uses LabelSpreading to get recommendations based on user likes and dislikes.

    Args:
        emb (np.ndarray): The embeddings matrix (n_samples, n_features).
        all_indices (list): List of all possible item indices (corresponding to rows in emb).

    Returns:
        tuple: (certainty_score, predicted_idx)
               certainty_score (float): Confidence score (0-100) for the prediction.
               predicted_idx (int or None): The index of the recommended item, or None if no recommendation.
    """
    pos_examples = st.session_state.choices.get("Thumbs Up", [])
    neg_examples = st.session_state.choices.get("Thumbs Down", [])

    # Check if there is any feedback to learn from
    if not pos_examples and not neg_examples:
        st.info("No feedback provided yet to train the recommendation model.")
        return 0.0, None # Cannot train without labels

    n_samples = emb.shape[0]
    if n_samples == 0:
        st.error("Cannot make recommendations: Embeddings data is empty.")
        return 0.0, None

    labels = np.full(n_samples, -1, dtype=int) # -1 for unlabeled

    # Assign labels based on user feedback, ensuring indices are valid
    valid_pos = [i for i in pos_examples if 0 <= i < n_samples]
    valid_neg = [i for i in neg_examples if 0 <= i < n_samples]

    if not valid_pos and not valid_neg:
         st.warning("No valid liked/disliked indices found within data bounds.")
         return 0.0, None

    labels[valid_pos] = 1  # 1 for liked items
    labels[valid_neg] = 0  # 0 for disliked items

    # Check if there are any unlabeled items left to predict
    if -1 not in labels:
        st.info("All items have been labeled (liked/disliked). No new items to recommend based on spreading.")
        return 0.0, None # Cannot spread if everything is labeled

    try:
        # Configure and run LabelSpreading
        # Parameters might need tuning based on dataset size and characteristics
        label_spread_model = LabelSpreading(kernel='knn',
                                            n_neighbors=10, # Affects locality
                                            alpha=0.2,      # Lower = more spreading influence from neighbors
                                            max_iter=1000,    # Increase if convergence issues
                                            tol=1e-4,       # Stricter tolerance
                                            n_jobs=-1)      # Use all CPU cores

        label_spread_model.fit(emb, labels)

        # Get the probability distribution over classes [dislike, like] (order depends on .classes_)
        probabilities = label_spread_model.label_distributions_

        # Find the column index corresponding to the 'like' class (label 1)
        like_class_index_tuple = np.where(label_spread_model.classes_ == 1)[0]

        if len(like_class_index_tuple) == 0:
            # This happens if only dislikes (label 0) were provided.
            st.info("Only 'dislike' feedback received. Cannot predict positive recommendations yet.")
            return 0.0, None # Cannot recommend a 'like' if the model didn't see any

        like_class_index = like_class_index_tuple[0]
        like_probabilities = probabilities[:, like_class_index] # Probabilities of being 'like'

        # Combine original indices with their 'like' probability
        # Ensure all_indices matches the length of like_probabilities
        if len(all_indices) != len(like_probabilities):
             st.error(f"Mismatch between number of indices ({len(all_indices)}) and probabilities ({len(like_probabilities)}).")
             return 0.0, None
        indexed_scores = list(zip(all_indices, like_probabilities))

        # Sort by probability (highest first)
        sorted_indexed_scores = sorted(indexed_scores, key=lambda item: item[1], reverse=True)

        # Filter out items already seen/interacted with
        # Pass the *original* all_indices list to filter_choices
        available_choices_indices = filter_choices(all_indices)
        available_choices_set = set(available_choices_indices) # Faster lookups

        # Find the best recommendation among available choices
        best_match_idx = None
        best_match_score = 0.0
        for idx, score in sorted_indexed_scores:
            # Check if the index is valid, available, and score is numeric
            if idx in available_choices_set and isinstance(score, (int, float)) and not np.isnan(score):
                best_match_idx = idx
                best_match_score = score
                break # Found the top available recommendation

        if best_match_idx is not None:
            # Scale score to 0-100 for consistency
            certainty_score = max(0.0, min(100.0, best_match_score * 100))
            return certainty_score, best_match_idx
        else:
            # No available choices left among the positively scored items
            st.info("No suitable new recommendations found based on current feedback and available items.")
            return 0.0, None

    except ValueError as ve:
         # Specific handling for common sklearn errors if needed
         st.error(f"Error during Label Spreading calculation: {ve}")
         # Could be due to input shapes, data types, etc.
         return 0.0, None
    except Exception as e:
        st.error(f"An unexpected error occurred during recommendation: {e}")
        import traceback
        st.error(traceback.format_exc()) # Print traceback for debugging
        return 0.0, None