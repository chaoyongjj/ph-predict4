import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt


# Load model
@st.cache_resource
def load_model():
    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


model = load_model()

# Create Streamlit app
st.title('Pulmonary Hypertension In-Hospital Mortality Prediction')

# Create input fields
st.header('Please enter patient information:')
feature_names = ['SAPSII', 'HR', 'PO2', 'Lactate', 'RDW']  # Replace with your feature names
features = {}

for feature in feature_names:
    features[feature] = st.number_input(f'Enter {feature}:', value=0.0)

# Create prediction button
if st.button('Predict'):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([features])

    # Make prediction
    prediction = model.predict_proba(input_df)[0][1]

    # Display prediction result
    st.subheader('Prediction Result:')
    st.write(f'In-hospital mortality probability: {prediction:.2%}')

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    print(shap_values)
    # Plot SHAP force plot
    st.subheader('SHAP Force Plot:')
#     fig, ax = plt.subplots()
    shap.force_plot(explainer.expected_value, shap_values[0], input_df,
                    feature_names=feature_names, matplotlib=True, show=False)
#     st.pyplot(fig)
    plt.savefig("force_plot.png", bbox_inches='tight', dpi=300)
    st.image("force_plot.png")
    # Plot SHAP bar chart
#     st.subheader('SHAP Feature Importance:')
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, input_df, plot_type="bar", feature_names=feature_names, show=False)
#     st.pyplot(fig)

# Add some explanatory information
st.markdown("""
### Instructions:
1. Enter the patient's information in the input fields above.
2. Click the "Predict" button to get the prediction results.
3. Review the predicted in-hospital mortality probability and SHAP explanation charts.

### Note:
- All input values should be numerical.
- Ensure the input data is within a reasonable range.
- The SHAP force plot shows the impact of each feature on the prediction outcome.
""")
