
import streamlit as st
import pandas as pd
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sales Predictor",
    page_icon="📈",
    layout="centered"
)

# --- MODEL LOADING ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model(model_path):
    """Loads the saved regression model from a .pkl file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

# Define the model filename
MODEL_FILENAME = 'model-reg-67130700364.pkl'
loaded_model = load_model(MODEL_FILENAME)


# --- APP LAYOUT ---
st.title("📈 Advertising Sales Predictor")
st.markdown("Enter the advertising budget for each channel to predict the total sales.")

# Display an error message if the model is not found
if loaded_model is None:
    st.error(
        f"**Model not found!** "
        f"Please make sure the model file (`{MODEL_FILENAME}`) is in the same directory as this script. "
        "You may need to run the model training script first."
    )
else:
    # --- USER INPUT ---
    with st.form("prediction_form"):
        st.subheader("Advertising Budgets ($)")

        # Create columns for a cleaner layout
        col1, col2, col3 = st.columns(3)

        with col1:
            # IMPORTANT: The input variable names must match the feature names the model was trained on.
            # Our model was trained on 'TV', 'radio', and 'newspaper'.
            youtube_budget = st.number_input("youtube", min_value=0, value=150, step=10, help="Budget for youtube")

        with col2:
            tiktok_budget = st.number_input("📻 tiktok", min_value=0, value=40, step=5, help="Budget for tiktok")

        with col3:
            instagram_budget = st.number_input("📰 instagram", min_value=0, value=60, step=5, help="Budget for instagram")

        # Submit button for the form
        submit_button = st.form_submit_button(label="Predict Sales")


    # --- PREDICTION AND OUTPUT ---
    if submit_button:
        # Create a DataFrame from the user inputs with the correct column names
        new_data = pd.DataFrame({
            'youtube': [youtube_budget],
            'tiktok': [tiktok_budget],
            'instagram': [instagram_budget]
        })

        st.markdown("---")
        st.subheader("Prediction Result")

        # Make a prediction
        try:
            prediction = loaded_model.predict(new_data)
            estimated_sales = prediction[0]

            # Display the result using st.metric for a nice visual
            st.metric(
                label="Predicted Sales",
                value=f"${estimated_sales:,.2f} K",
                help="This is the estimated total sales based on the provided ad budgets."
            )

            # Add a success message
            st.success("Prediction was successful!")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- SIDEBAR INFORMATION ---
st.sidebar.header("About This App")
st.sidebar.info(
    "This application uses a Linear Regression model to predict sales based on advertising spending. "
    "The model was trained on the 'advertising.csv' dataset."
)
st.sidebar.markdown(
    "**Note:** The model was trained with features named `youtube`, `tiktok`, and `instagram`. "
    "The input fields correspond to these original features."
)
