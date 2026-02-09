# app/app.py
# Streamlit House Price Prediction Web App (LKR + no negatives)
# This application provides a web interface for predicting house prices using a machine learning model
# Prices are displayed in Sri Lankan Rupees (LKR) with validation to prevent negative predictions

# Import future annotations to allow type hints using standard collection types
from __future__ import annotations

from dataclasses import dataclass       # Import dataclass decorator for creating simple data container classes
from pathlib import Path                # Import Path class for cross-platform file path operations
from typing import Any, Dict, List      # Import type hints for better code documentation and IDE support


import joblib   # Import joblib for loading the serialized machine learning model
import matplotlib.pyplot as plt  # Import matplotlib for creating data visualizations and graphs
import numpy as np    # Import numpy for numerical operations and array manipulations
import pandas as pd  # Import pandas for data manipulation and DataFrame operations
import streamlit as st  # Import streamlit for building the web application interface


# ----------------------------- CONFIG -----------------------------
# Configure the Streamlit page settings with a custom title and wide layout
st.set_page_config(page_title="House Price Prediction", layout="wide")

# Get the absolute path to the project root directory (parent of app directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Define the path where the trained ML model is stored
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_pipeline.joblib"

# Set hardcoded credentials for simple authentication (not secure for production)
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

# Currency conversion rate: 1 USD = 300 LKR (Sri Lankan Rupees)
LKR_PER_USD = 300.0

# Dictionary containing default values for all input fields in the form
DEFAULTS = {
    "gr_liv_area_val": 1500,      # Default living area: 1500 square feet
    "bedrooms_val": 3,             # Default number of bedrooms: 3
    "full_bath_val": 2,            # Default number of full bathrooms: 2
    "overall_qual_val": 5,         # Default quality rating: 5 (out of 10)
    "year_built_val": 2000,        # Default construction year: 2000
}


# ----------------------------- HELPERS -----------------------------
# Decorator to cache the model loading function - model loads only once and is reused
@st.cache_resource
def load_model(model_path: Path):
    """
    Load the trained machine learning model from disk.
    
    Args:
        model_path: Path object pointing to the saved model file
        
    Returns:
        Loaded scikit-learn pipeline object
        
    Raises:
        FileNotFoundError: If the model file doesn't exist at the specified path
    """
    # Check if the model file exists before attempting to load
    if not model_path.exists():
        # Raise an error with a descriptive message if file is not found
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
        )
    # Load and return the model using joblib (efficient for scikit-learn objects)
    return joblib.load(model_path)


def get_expected_columns(pipeline) -> List[str]:
    """
    Extract the list of feature column names expected by the ML pipeline.
    
    Args:
        pipeline: Scikit-learn pipeline object containing the trained model
        
    Returns:
        List of column names that the model expects as input features
    """
    # Access the preprocessor step from the pipeline
    pre = pipeline.named_steps["preprocessor"]
    # Return the feature names the preprocessor was trained on
    return list(pre.feature_names_in_)


def make_input_df(expected_cols: List[str], values: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with the correct column structure for model prediction.
    
    Args:
        expected_cols: List of column names the model expects
        values: Dictionary mapping column names to their values
        
    Returns:
        Single-row DataFrame ready for model prediction
    """
    # Initialize a dictionary with all expected columns set to None
    row = {c: None for c in expected_cols}
    # Iterate through provided values and update the row dictionary
    for k, v in values.items():
        # Only add values for columns that the model expects
        if k in row:
            row[k] = v
    # Convert the dictionary to a DataFrame with a single row and return
    return pd.DataFrame([row])


# Define a dataclass to hold all user input values in a structured way
@dataclass
class AppInputs:
    """
    Data container for storing house feature inputs from the user.
    Using dataclass for cleaner code and automatic __init__ generation.
    """
    gr_liv_area: int      # Above ground living area in square feet
    bedrooms: int          # Number of bedrooms above ground
    full_bath: int         # Number of full bathrooms
    overall_qual: int      # Overall quality rating (1-10)
    year_built: int        # Year the house was built


def init_session_state() -> None:
    """
    Initialize Streamlit session state variables on first app load.
    Session state persists data across reruns and user interactions.
    """
    # Initialize login status if not already set (user starts logged out)
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Initialize empty list to store prediction history records
    if "records" not in st.session_state:
        st.session_state.records = []

    # Set default values for all input fields using the DEFAULTS dictionary
    for k, v in DEFAULTS.items():
        # Only initialize if the key doesn't already exist in session state
        if k not in st.session_state:
            st.session_state[k] = v

    # Initialize variable to store the most recent prediction result
    if "last_prediction_lkr" not in st.session_state:
        st.session_state.last_prediction_lkr = None


def clear_inputs() -> None:
    """
    Reset all input field values back to their defaults.
    This clears the form but preserves the prediction history.
    """
    # Loop through defaults and reset each value in session state
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    # Clear the last prediction to remove the "Add to History" button
    st.session_state.last_prediction_lkr = None


def clear_history() -> None:
    """
    Clear only the prediction history and graphs.
    Input fields remain unchanged.
    """
    # Reset the records list to empty, removing all historical predictions
    st.session_state.records = []


def do_logout() -> None:
    """
    Log out the user and reset all application state.
    This is a complete reset of the application.
    """
    # Set login status to False to return user to login page
    st.session_state.logged_in = False
    # Clear all prediction history
    st.session_state.records = []
    # Reset input fields to defaults
    clear_inputs()


# ----------------------------- UI PAGES -----------------------------
def login_page() -> None:
    """
    Render the login page UI with username/password authentication.
    This is the first page users see before accessing the main application.
    """
    # Display the main title of the application
    st.title("🏠 House Price Prediction")
    # Add a subtitle/caption below the title
    st.caption("Login to continue")

    # Display a decorative house icon image from an external URL
    st.image("https://cdn-icons-png.flaticon.com/512/69/69524.png", width=140)

    # Create text input field for username (not password-protected)
    username = st.text_input("Username")
    # Create password input field (text is hidden with asterisks)
    password = st.text_input("Password", type="password")

    # Create a login button that spans the full width of the container
    if st.button("Login", use_container_width=True):
        # Check if entered credentials match the valid credentials
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            # Set logged_in flag to True in session state
            st.session_state.logged_in = True
            # Show success message to user
            st.success("✅ Login successful")
            # Rerun the app to redirect to main page
            st.rerun()
        else:
            # Show error message if credentials are invalid
            st.error("❌ Invalid username or password")


def main_page(model) -> None:
    """
    Render the main application page with input form, prediction, and history.
    
    Args:
        model: The loaded machine learning model pipeline
    """
    # Display the main page title
    st.title("🏠 House Price Prediction App")

    # Get the list of column names the model expects
    expected_cols = get_expected_columns(model)

    # Display section header for input form
    st.subheader("Enter House Details")
    # Create three equal-width columns for organizing input fields
    c1, c2, c3 = st.columns(3)

    # First column: Living Area and Bedrooms
    with c1:
        # Number input for living area in square feet
        st.number_input(
            "Living Area (sqft)",          # Label displayed to user
            min_value=100,                  # Minimum allowed value
            max_value=10000,                # Maximum allowed value
            step=10,                        # Increment/decrement step
            value=int(st.session_state.gr_liv_area_val),  # Current value from session
            key="gr_liv_area_widget",       # Unique key for this widget
        )

        # Number input for number of bedrooms
        st.number_input(
            "Bedrooms",
            min_value=0,                    # Allow 0 bedrooms (studio apartment)
            max_value=20,                   # Maximum bedrooms
            step=1,                         # Increment by 1
            value=int(st.session_state.bedrooms_val),
            key="bedrooms_widget",
        )

    # Second column: Bathrooms and Quality
    with c2:
        # Number input for number of full bathrooms
        st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=10,
            step=1,
            value=int(st.session_state.full_bath_val),
            key="full_bath_widget",
        )

        # Number input for overall quality rating (1-10 scale)
        st.number_input(
            "Overall Quality (1–10)",
            min_value=1,                    # Minimum quality rating
            max_value=10,                   # Maximum quality rating
            step=1,
            value=int(st.session_state.overall_qual_val),
            key="overall_qual_widget",
        )

    # Third column: Year Built
    with c3:
        # Number input for year the house was constructed
        st.number_input(
            "Year Built",
            min_value=1800,                 # Earliest reasonable construction year
            max_value=2026,                 # Current year + 1 for new construction
            step=1,
            value=int(st.session_state.year_built_val),
            key="year_built_widget",
        )

    # Sync widget values to safe session keys
    # This stores the current widget values in separate session variables
    # to prevent conflicts when clearing inputs
    st.session_state.gr_liv_area_val = st.session_state.gr_liv_area_widget
    st.session_state.bedrooms_val = st.session_state.bedrooms_widget
    st.session_state.full_bath_val = st.session_state.full_bath_widget
    st.session_state.overall_qual_val = st.session_state.overall_qual_widget
    st.session_state.year_built_val = st.session_state.year_built_widget

    # Create three columns for action buttons
    btn1, btn2, btn3 = st.columns(3)

    # First button column: Predict Price button
    with btn1:
        # Create predict button and capture whether it was clicked
        predict_clicked = st.button("Predict Price", use_container_width=True)

    # Second button column: Clear Inputs button
    with btn2:
        # Button that calls clear_inputs function when clicked
        st.button("Clear Inputs", use_container_width=True, on_click=clear_inputs)

    # Third button column: Logout button
    with btn3:
        # Button that calls do_logout function when clicked
        st.button("Logout", use_container_width=True, on_click=do_logout)

    # Safety check: if user logged out, rerun to show login page
    if not st.session_state.logged_in:
        st.rerun()

    # Handle prediction when Predict Price button is clicked
    if predict_clicked:
        # Create dictionary mapping model column names to current input values
        values = {
            "GrLivArea": st.session_state.gr_liv_area_val,       # Living area
            "BedroomAbvGr": st.session_state.bedrooms_val,       # Bedrooms
            "FullBath": st.session_state.full_bath_val,          # Bathrooms
            "OverallQual": st.session_state.overall_qual_val,    # Quality
            "YearBuilt": st.session_state.year_built_val,        # Year built
        }

        # Create a DataFrame in the format expected by the model
        input_df = make_input_df(expected_cols, values)

        # Try to make prediction and handle any errors gracefully
        try:
            # Get prediction from model (returns log-transformed value)
            pred_log = float(model.predict(input_df)[0])
            # Reverse log transformation to get actual USD price
            pred_usd = float(np.expm1(pred_log))
            # Convert USD to LKR using the conversion rate
            pred_lkr = pred_usd * LKR_PER_USD

            # Ensure prediction is non-negative (floor at 0)
            pred_lkr = max(pred_lkr, 0.0)

            # Store prediction in session state for use by "Add to History" button
            st.session_state.last_prediction_lkr = pred_lkr
            # Display success message with formatted price
            st.success(f"💰 Predicted Price: LKR {pred_lkr:,.2f}")

        # Catch any errors during prediction (e.g., invalid input, model issues)
        except Exception as e:
            # Show generic error message to user
            st.error("Prediction failed")
            # Display the full exception details for debugging
            st.exception(e)

    # Show "Add to History" button only if a prediction was just made
    if st.session_state.last_prediction_lkr is not None:
        # Create button to save current prediction to history
        if st.button("Add to History"):
            # Append a new record dictionary to the records list
            st.session_state.records.append(
                {
                    "GrLivArea": st.session_state.gr_liv_area_val,
                    "Bedrooms": st.session_state.bedrooms_val,
                    "Bathrooms": st.session_state.full_bath_val,
                    "Quality": st.session_state.overall_qual_val,
                    "YearBuilt": st.session_state.year_built_val,
                    "PredictedPrice_LKR": st.session_state.last_prediction_lkr,
                }
            )
            # Show confirmation message
            st.success("Added to history")

    # Add visual separator between sections
    st.divider()
    # Display section header for history and visualizations
    st.subheader("History & Graphs")

    # Check if there are any saved prediction records
    if st.session_state.records:
        # Convert list of dictionaries to pandas DataFrame for display
        hist_df = pd.DataFrame(st.session_state.records)
        # Display the DataFrame as an interactive table
        st.dataframe(hist_df)

        # Create two columns for side-by-side charts
        colA, colB = st.columns(2)

        # First chart column: Line plot of predictions over time
        with colA:
            # Create matplotlib figure and axis objects
            fig1, ax1 = plt.subplots()
            # Plot predicted prices as a line graph with markers
            ax1.plot(hist_df["PredictedPrice_LKR"], marker="o")
            # Set x-axis label
            ax1.set_xlabel("Record #")
            # Set y-axis label
            ax1.set_ylabel("Price (LKR)")
            # Display the matplotlib figure in Streamlit
            st.pyplot(fig1)

        # Second chart column: Scatter plot of price vs. living area
        with colB:
            # Create new matplotlib figure and axis
            fig2, ax2 = plt.subplots()
            # Create scatter plot showing relationship between area and price
            ax2.scatter(hist_df["GrLivArea"], hist_df["PredictedPrice_LKR"])
            # Set x-axis label
            ax2.set_xlabel("Living Area")
            # Set y-axis label
            ax2.set_ylabel("Price (LKR)")
            # Display the scatter plot
            st.pyplot(fig2)

        # Button to clear all prediction history
        st.button("Clear History", on_click=clear_history)
    else:
        # Show informational message when no records exist yet
        st.info("No records yet.")


# ----------------------------- APP ENTRY -----------------------------
def main():
    """
    Main entry point of the application.
    Initializes session state, loads the model, and routes to appropriate page.
    """
    # Initialize all session state variables (runs on first load)
    init_session_state()

    # Load the trained machine learning model from disk (cached)
    model = load_model(MODEL_PATH)

    # Route to appropriate page based on login status
    if st.session_state.logged_in:
        # User is logged in: show main application page
        main_page(model)
    else:
        # User is not logged in: show login page
        login_page()


# Standard Python idiom: only run main() if this file is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main() 