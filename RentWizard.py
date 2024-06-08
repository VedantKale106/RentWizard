import pickle 
import streamlit as st
import numpy as np

def load_model():
    with open('Model.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

forest = data["model"]
le_furnishing = data["le_furnishing"]
le_available_for = data["le_available_for"]

def show_predict_page():

    st.image("img.png") 
    st.title("Rent Predictor in Pune")
    st.write("### We need some information to predict the rent")

    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            right: 10px;
            bottom: 10px;
            font-size: 12px;
            color: gray;
        }
        </style>
        <div class="footer">
        Made by Vedant 
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("Fill out the details below and click 'Calculate Rent' to get an estimate.")

    furnishing = ("Furnished", "Semifurnished", "Unfurnished")
    available_for = ("Bachelors", "Family", "All")

    furnishing = st.selectbox("Select Furnishing Status", furnishing)
    available_for = st.selectbox("Select Availability", available_for)

    col1, col2 = st.columns(2)

    with col1:
        rooms = st.slider("Number of Rooms", 0, 8, 2)
        bathrooms = st.slider("Number of Bathrooms", 0, 5, 2)

    with col2:
        area = st.slider("Area in sqft", 0, 3000, 1000)

    ok = st.button("Calculate Rent")

    if ok:
        X = np.array([[rooms, bathrooms, area, furnishing, available_for]])
        X[:, 3] = le_furnishing.transform(X[:, 3])
        X[:, 4] = le_available_for.transform(X[:, 4])
        X = X.astype(float)

        rent = forest.predict(X)
        st.subheader(f"The Estimated Rent is {rent[0]:.2f} Rs per Month")

def show_sidebar():
    st.sidebar.title("Contact Details")
    st.sidebar.info(
        """
        Vedant Kale  
        Email: vedant.kale22@pccoepune.org
        Phone: +91-8421204009             
        LinkedIn: [Vedant Kale](https://www.linkedin.com/in/vedantkale106/)
        """
    )

show_sidebar()
show_predict_page()
