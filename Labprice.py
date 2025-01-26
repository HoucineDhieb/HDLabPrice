# %%
# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import webbrowser  # To open the profiling report in the browser

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ydata_profiling import ProfileReport


# %%
# Function: Load Dataset
rf_model = None  # Initialize Random Forest model variable
@st.cache_data
def load_dataset(filepath):
    """Load dataset from a given filepath."""
    # Ensure the dataset file exists in the project folder
    df = pd.read_csv(filepath)
    return df


# Function: Cached Generate Profiling Report
@st.cache_data
def generate_profiling_report(dataset):
    """Generate Pandas Profiling Report and save it as an HTML file."""
    profile = ProfileReport(dataset, title="Price Prediction Dataset Report", explorative=True)
    report_file = "profiling_report.html"
    profile.to_file(report_file)
    return report_file


# %%
# Function: Preprocessing Pipeline
def preprocess_pipeline():
    """Build a preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())  # Standardize numeric features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, ['Indice Prix','Week Number','Year','Weekly Volume','Prix H.T €/l','Value / T','Packaging']),  # Adjust for numeric column names in your data
        ('cat', categorical_transformer, ['Week', 'Article','Brand','Oil Type'])  # Replace with actual columns
    ])

    return preprocessor


# %%
# Function: Train Model
def train_model(train_X, test_X, train_y, test_y, model, model_name="Model"):
    """
    Train a model and evaluate its performance.
    """
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)

    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)

    # Report performance
    st.write(f"### {model_name} Performance")
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R-squared (R2):", r2)
    return model, mse, r2


# %%
# Function: Cross-Validation
def perform_cross_validation(model, X, y):
    """
    Perform cross-validation on a given model and dataset.
    """
    scores = cross_val_score(model, X, y, cv=5)
    st.write("Cross-validation scores:", scores)
    st.write("Mean cross-validation score:", scores.mean())
    return scores


# %%
# Function: Hyperparameter Tuning
def tune_parameters(model, param_grid, X_train, y_train):
    """
    Perform a GridSearchCV to tune hyperparameters.
    """
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    st.write("### Grid Search Best Parameters")
    best_params = grid_search.best_params_
    st.write("Best Parameters:", best_params)
    return grid_search.best_estimator_


# %%
# Streamlit App Starts
st.title("Price Prediction App")

# Load dataset
filepath = st.text_input("Enter the dataset file path (CSV)", value="Price_Prediction_Project_Dataset.csv")
if filepath:
    dataset = load_dataset(filepath)
    st.write("### Dataset Loaded Successfully")
    st.dataframe(dataset.head())

    # Profiling report
    st.write("### Dataset Profiling Report")
    if st.button("Generate Profiling Report"):
        # Generate and cache the profiling report when button is clicked
        report_file = generate_profiling_report(dataset)

        # Open the report in the browser
        webbrowser.open_new_tab(report_file)

        # Notify the user
        st.write(f"Profiling report saved as '{report_file}' and opened in the browser!")
        st.write(f"Manually open the report by downloading or opening: `{report_file}`")

    # Check missing data
    st.write("### Handle Missing Data")
    if st.checkbox("Show Missing Data Summary"):
        st.write(dataset.isnull().sum())
    if st.checkbox("Fill Missing Values with Mean"):
        dataset['Indice Prix'].fillna(dataset['Indice Prix'].mean(), inplace=True)
        st.write("Missing values handled.")

    # Pairplot
    st.write("### Pairplot of Dataset")
    if st.checkbox("Show Pairplot"):

        pairplot = sns.pairplot(dataset)

        st.pyplot(pairplot)

    # Dataset Splitting
    st.write("### Data Preparation")
    if st.checkbox("Split Dataset for Training"):
        if 'Price' in dataset.columns:
            X = dataset.drop('Price', axis=1)
            y = dataset['Price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.write("Data split into training and testing sets.")
        else:
            st.write("Error: No column named 'Price' in the dataset!")

    # Choose a model
    st.write("### Choose a Model")
    model_option = st.selectbox("Select Model", ["CatBoost Regressor", "Random Forest Regressor"])
    st.write("### Set Model Parameters")
    if model_option == "CatBoost Regressor":
        iterations = st.number_input("Iterations", min_value=1, value=100, step=10)
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        depth = st.slider("Depth", min_value=1, max_value=16, value=6, step=1)
    elif model_option == "Random Forest Regressor":
        n_estimators = st.number_input("Number of Estimators", min_value=1, value=100, step=10)
        max_depth = st.number_input("Max Depth (None for no limit)", min_value=1, value=10, step=1)
        min_samples_split = st.number_input("Min Samples Split", min_value=2, value=2)

    preprocessor = preprocess_pipeline()  # Get preprocessing pipeline
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    if model_option == "CatBoost Regressor":
        if 'catboost_model' not in st.session_state:
            st.session_state['catboost_model'] = None

        if st.button("Train CatBoost Model"):
            st.session_state['catboost_model'] = CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0)
            trained_model, mse, r2 = train_model(X_train, X_test, y_train, y_test, st.session_state['catboost_model'], "CatBoost Regressor")

        if st.checkbox("Perform Cross Validation"):
            if st.session_state['catboost_model'] is not None:
                perform_cross_validation(st.session_state['catboost_model'], X_train, y_train)
            else:
                st.error("Please train the CatBoost model first.")

        if st.button("Perform GridSearch"):
            if st.session_state['catboost_model'] is not None:
                param_grid = {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.1],
                'depth': [4, 6, 8]
            }
                best_model = tune_parameters(st.session_state['catboost_model'], param_grid, X_train, y_train)
            else:
                st.error("Please train the CatBoost model first.")

    elif model_option == "Random Forest Regressor":
        if 'rf_model' not in st.session_state:
            st.session_state['rf_model'] = None

        if st.button("Train Random Forest Model"):
            st.session_state['rf_model'] = RandomForestRegressor(n_estimators=n_estimators, max_depth=None if max_depth == 0 else max_depth, min_samples_split=min_samples_split, random_state=42)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            trained_model, mse, r2 = train_model(X_train, X_test, y_train, y_test, st.session_state['rf_model'], "Random Forest Regressor")

        if st.checkbox("Perform Cross Validation"):
            if st.session_state['rf_model'] is not None:

                perform_cross_validation(st.session_state['rf_model'], X_train, y_train)
            else:
                st.error("Please train the Random Forest model first.")

        if st.button("Perform GridSearch"):
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }
            if st.session_state['rf_model'] is not None:

                best_model = tune_parameters(st.session_state['rf_model'], param_grid, X_train, y_train)
            else:
                st.error("Please train the Random Forest model first.")
