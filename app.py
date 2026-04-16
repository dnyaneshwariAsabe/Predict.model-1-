import streamlit as st
import pickle
import numpy as np
import time

# --- Page Config ---
st.set_page_config(page_title="Customer Classifier", page_icon="🎯", layout="centered")

# --- Custom CSS for Animation & Styling ---
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-container {
        animation: fadeIn 1s ease-out;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(45deg, #4CAF50, #2E7D32);
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        with open('model (1).pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- App UI ---
st.title("🎯 Customer Purchase Predictor")
st.write("Enter the customer details below to predict their purchase likelihood.")

with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 30)
        
    with col2:
        salary = st.number_input("Estimated Annual Salary ($)", min_value=0, value=50000, step=1000)

    # Convert Gender to numeric (0 for Female, 1 for Male)
    # Note: If your model was trained on Female=0/Male=1, this will work. 
    # If it used One-Hot Encoding, you may need to adjust this logic.
    gender_val = 1 if gender == "Male" else 0
    
    features = np.array([[gender_val, age, salary]])

    if st.button("Analyze Prediction"):
        if model:
            with st.spinner('Calculating probabilities...'):
                time.sleep(1) # Visual effect for "animation"
                prediction = model.predict(features)
                
                st.divider()
                if prediction[0] == 1:
                    st.balloons()
                    st.success("### result: Likely to Purchase! ✅")
                else:
                    st.warning("### Result: Unlikely to Purchase. ❌")
        else:
            st.error("Model file not found. Please ensure 'model (1).pkl' is in the same directory.")
    
    st.markdown('</div>', unsafe_allow_html=True)
