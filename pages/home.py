import streamlit as st

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;font-weight: bold; color:#00B2A9'>Welcome to my Credit App</h1>", unsafe_allow_html=True)

st.write("#")

st.write(
    """
This project involves an application that predicts which clients are likely to experience credit delinquency within the next two years using a pre-trained Machine Learning model. 
It also offers insights into why clients may be at risk of delinquency and allows for model retraining to adapt to changing market conditions.\n
Please read the following sections to understand all the features of the solution.
    """ 
)
st.write("#")
st.write("### 1. Predict Page")
st.write(
    """    
This feature allows users to predict credit delinquency for multiple clients simultaneously. Indeed, the user uploads a CSV or TXT file containing client data (features), and the application uses the saved Machine Learning model to predict which clients are likely to be in delinquency on their credit (1 if yes 0 otherwise).\n
The user can decide:
- Return or not input data.
- Calculate probability of each class.
    """
)

st.write("#")
st.write("### 2. Insights Page")

st.write(
    """    
This functionality provides users with explanations for the predictions made by the Machine Learning model, helping them understand the reasons behind each prediction and the factors contributing to the likelihood of credit delinquency. 
\nThe insights are divided into two main sections:
- **Feature Importance**
    - **Split Importance**: Measures how often a feature is used in the decision trees. The more frequent the splits, the higher the importance.
    - **Gain Importance**: Measures the improvement in accuracy brought by a feature. Higher gains indicate higher importance.

These importances help in understanding which features are most influential in making predictions, aiding in model interpretability and feature selection.
#
- **SHAP Plots**
    - **Summary Plot**: Visualizes the impact of features on model predictions using Shapley values. Each point represents a Shapley value for a feature, with colors indicating feature values (e.g., red for high, blue for low). 
    This plot helps understand which features most influence the model's predictions and their effect.
    - **Dependence Plot**: Shows the relationship between a single feature's value and its corresponding SHAP values, indicating how changes in the feature impact model predictions. 
    It helps identify non-linear effects and interactions between features, providing deeper insights into the model's behavior.
    """
)

st.write("#")
st.write("### 3. Train Page")
st.write(
    """    
This feature allows users to retrain a new ML model in case of changes in market behavior. Users can select new features to create a more accurate model. \n
After training, users can evaluate the new model by comparing its metrics with historical models using displayed plots. \n
If satisfied, users can save the model, making it the default for the app; otherwise, they can cancel, and no new model will be created.
    """
)
#with st.expander("⚙️ - How to use it ", expanded=False):



