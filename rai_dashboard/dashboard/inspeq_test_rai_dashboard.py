import streamlit as st
import requests
import pandas as pd
import streamlit.components.v1 as components

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000/evaluate/"

# Streamlit UI
st.title("Responsible AI Metrics Dashboard")

# Step 1: Allow user to choose the section to display
tabs = ["RAI Metrics", "Data Analysis"]  # Tabs for navigation
selected_tab = st.sidebar.radio("Select a Section", tabs)

if selected_tab == "RAI Metrics":
    # Step 2: Allow user to load or input test data
    data_option = st.radio("Choose input method", ("Upload Data", "Input Text"))

    if data_option == "Upload Data":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            # Load the data
            data = pd.read_csv(uploaded_file)
            st.write("Data preview:", data.head())

            # Extract the text and labels (adjust column names)
            X_test = data['text_column'].values  # Adjust column name as per your dataset
            y_test = data['label_column'].values  # Adjust target column name
            st.write("Test data loaded successfully!")

            # Step 3: Send data to FastAPI for evaluation
            response = requests.post(FASTAPI_URL, json={"text": X_test.tolist()})

            if response.status_code == 200:
                metrics = response.json()
                st.subheader("📊 RAI Metrics")
                st.write(f"Precision: {metrics['precision']}")
                st.write(f"Recall: {metrics['recall']}")
                st.write(f"F1 Score: {metrics['f1_score']}")
                st.write(f"ROC AUC: {metrics['roc_auc']}")
                st.write(f"Brier Score: {metrics['brier_score']}")
                st.write(f"Average Confidence: {metrics['avg_confidence']}")

                st.subheader("🌀 Perturbation Sensitivity Score")
                col7, col8, col9 = st.columns(3)
                col7.metric("PSS Score", metrics["pss_score"])
                col8.metric("Original Probability", metrics["original_prob"])
                col9.metric("Perturbation Std Dev", metrics["augmented_std"])

                st.subheader("🔁 Perturbed Texts Used for PSS")
                for i, aug_text in enumerate(metrics["perturbed_texts"], 1):
                    st.markdown(f"**Perturbation {i}:** {aug_text}")
            else:
                st.error("Error fetching metrics from FastAPI.")
    
    elif data_option == "Input Text":
        default_text = "Pharmaceuticals group Orion Corp reported a fall in its third-quarter earnings that were hit by larger expenditures on R&D and marketing"
        text_input = st.text_area("Enter your text for evaluation:", value=default_text)
        if st.button("Evaluate"):
            if text_input:
                # Step 3: Send data to FastAPI for evaluation
                response = requests.post(FASTAPI_URL, json={"text": text_input})  # Send as string

                if response.status_code == 200:
                    metrics = response.json()
                    st.subheader("📊 RAI Metrics")
                    st.write(f"Precision: {metrics['precision']}")
                    st.write(f"Recall: {metrics['recall']}")
                    st.write(f"F1 Score: {metrics['f1_score']}")
                    st.write(f"ROC AUC: {metrics['roc_auc']}")
                    st.write(f"Brier Score: {metrics['brier_score']}")
                    st.write(f"Average Confidence: {metrics['avg_confidence']}")

                    st.subheader("🌀 Perturbation Sensitivity Score")
                    col7, col8, col9 = st.columns(3)
                    col7.metric("PSS Score", metrics["pss_score"])
                    col8.metric("Original Probability", metrics["original_prob"])
                    col9.metric("Perturbation Std Dev", metrics["augmented_std"])

                    st.subheader("🔁 Perturbed Texts Used for PSS")
                    for i, aug_text in enumerate(metrics["perturbed_texts"], 1):
                        st.markdown(f"**Perturbation {i}:** {aug_text}")
                else:
                    st.error("Error fetching metrics from FastAPI.")
            else:
                st.warning("Please enter some text to evaluate.")

elif selected_tab == "Data Analysis":
    # Step 4: Load and display the HTML content in the "Data Analysis" tab with scrolling enabled

    try:
        with open("/Users/piyush/Desktop/Codes/inspeq_casestudy/rai_dashboard/models/adult_income_profile.html", "r") as f:
            html_content = f.read()

        # Embed HTML content with scrolling enabled
        components.html(html_content, height=800, width=800, scrolling=True)
    except Exception as e:
        st.error(f"Failed to load HTML file: {str(e)}")
