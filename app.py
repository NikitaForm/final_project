from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import plotly.graph_objects as go
from model import MODELS, model_filename

def load_and_predict(X: ArrayLike, filename: str = "linear_regression.joblib") -> ArrayLike:
    """
    Deserialize and load the regression model and use it to predict on user provided data.

    This function takes a file name 'filename' that has a default value.
    It uses Joblib 'load' to load the model using the provided file name.
    When the model is loaded, call its `predict` method on provied data.

    Args:
        X (array-like): User provided data used for prediction.
        filename (str): Name of the file that is used to store the model.

    Returns:
        np.ndarray: Predicted value.
    """
    
    model = load(filename)
    y = model.predict(X)

    return y

def create_streamlit_app():
    """
    Creates a Streamlit web application for making predictions with a simple regression model.

    This function sets up a Streamlit app with a user interface for inputting a single feature 
    value and making predictions using a pre-trained regression model. The app includes:
    
    - A title displayed at the top of the app.
    - A slider for the user to select an input feature value within a specified range (-3.0 to 3.0).
    - A "Predict value" button that, when clicked, triggers the prediction process.
    - Upon clicking the "Predict value" button, the function:
        - Calls `load_and_predict`, passing the selected feature as input, to load the regression model 
          and make a prediction.
        - Displays the prediction result on the app.
        - Calls `visualize_difference`, passing the input feature and the prediction result, 
          to visualize the difference between the predicted value and the actual value in the original dataset.

    Note: This function does not return any value. It directly manipulates the Streamlit app's UI by 
    writing content and rendering UI elements.
    """
    st.title("Regression Predictor")

    model_name = st.selectbox("Model", list(MODELS.keys()))

    filename = model_filename(model_name)
    model = load(filename)
    metrics = load("metrics.joblib")[model_name]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R2 Score", f"{metrics['r2']:.3f}")
    col2.metric("MSE", f"{metrics['mse']:.1f}")
    col3.metric("Slope", f"{float(model.coef_[0]):.3f}")
    col4.metric("Intercept", f"{float(model.intercept_):.3f}")

    input_feature = st.slider("Input feature", -3.0, 3.0, 0.0)

    if st.button("Predict value"):
        prediction = load_and_predict([[input_feature]], filename)

        st.write(f"Prediction: {prediction[0]:.3f}")

        visualize_difference(input_feature, prediction[0], model)

def visualize_difference(input_feature: float, prediction: ArrayLike, model):
    """
    Deserialize and load the initial datasets. Calculate the difference between actual data
    in the 'y' dataset and the predicted value for a given 'input_feature'.

    Visualize the difference by plotting the entire 'X' & 'y' as a Scatter plot. Then add
    a blue dot that represents the actual target value, and a red dot that represents the predicted target value for the given 'input_feature'.
    Add a dashed line connects these points, highlighting the difference between them, which is annotated on the plot.

    Args:
        input_feature (float): User provided data used for prediction.
        prediction (array-like): Predicted value.
        model: Trained regression model used to draw the regression line and residuals.

    """
    X_filename = "X.joblib"
    y_filename = "y.joblib"

    X = load(X_filename)

    y = load(y_filename)

    actual_target = y[_index_of_closest(X, input_feature)]

    difference = float(actual_target - prediction)

    X_flat = np.asarray(X).ravel()
    order = np.argsort(X_flat)
    line_x = X_flat[order]
    line_y = model.predict(X_flat.reshape(-1, 1))[order]

    is_dark = st.context.theme.type == "dark"
    line_color = "white" if is_dark else "black"
    template = "plotly_dark" if is_dark else "plotly"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_flat, y=np.asarray(y), mode="markers", name="Dataset", marker=dict(color="lightgrey")))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Regression line", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=[input_feature], y=[float(actual_target)], mode="markers", name="Actual target", marker=dict(color="blue", size=12)))
    fig.add_trace(go.Scatter(x=[input_feature], y=[float(prediction)], mode="markers", name="Predicted target", marker=dict(color="red", size=12)))
    fig.add_trace(go.Scatter(x=[input_feature, input_feature], y=[float(actual_target), float(prediction)], mode="lines", line=dict(color=line_color, dash="4px,2px"), showlegend=False))
    fig.add_annotation(x=input_feature, y=(float(actual_target) + float(prediction)) / 2, text=f"{difference:.2f}", showarrow=False, xshift=20)
    fig.update_layout(title="Difference between actual and predicted target", xaxis_title="Feature", yaxis_title="Target", template=template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(hoverformat=".2f"), yaxis=dict(hoverformat=".2f"))

    st.plotly_chart(fig, use_container_width=True)

# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    """
    This function takes an array-like object `X` and a float `k`, and returns the index of the 
    element in `X` that is closest to `k`. The function first converts `X` into a NumPy array 
    (if it isn't one already) to ensure compatibility with NumPy operations. It then calculates 
    the absolute difference between each element in `X` and `k`, identifies the minimum value 
    among these differences, and returns the index of this minimum difference.

    Args:
        X (ArrayLike): An array-like object containing numerical data. It can be a list, tuple, 
      or any object that can be converted to a NumPy array.
        k (float): The target value to which the closest element in `X` is sought.

    Returns:
        int: The index of the element in `X` that is closest to the value `k`.
    Returns:
        int: Index for the closest value to k in X.
    Finds the index of the element in `X` that is closest to the value `k`.

    """
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()