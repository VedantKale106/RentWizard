import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Load the trained model and encoders
def load_model():
    with open('Model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load data
data = load_model()
forest = data["model"]
le_furnishing = data["le_furnishing"]
le_available_for = data["le_available_for"]

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #34495e;
        margin-bottom: 30px;
    }
    .footer {
        position: fixed;
        bottom: 10px;
        right: 20px;
        font-size: 14px;
        color: #7f8c8d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ecf0f1;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3498db;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with contact details and insights
def show_sidebar():
    st.sidebar.title("📞 Contact & Insights")
    
    # Contact Details
    st.sidebar.header("👤 Developer")
    st.sidebar.info(
        """
        **Vedant Kale**  
        📧 Email: [vedant.kale22@pccoepune.org](mailto:vedant.kale22@pccoepune.org)  
        📱 Phone: +91-8421204009  
        🔗 [LinkedIn](https://www.linkedin.com/in/vedantkale106/)  
        """
    )
    
    # Pune Real Estate Insights
    st.sidebar.header("🏘️ Pune Real Estate Insights")
    insights = [
        "Pune's rental market is dynamic and growing",
        "IT hubs influence rental prices significantly",
        "Areas like Koregaon Park command premium rents",
        "Proximity to tech parks affects property values"
    ]
    for insight in insights:
        st.sidebar.markdown(f"• {insight}")

# Rent Prediction Page
def show_predict_page():
    st.image("img.png", width=600)
    st.markdown('<p class="main-title">🏡 Pune Rent Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Estimate Your Property\'s Rental Value with Machine Learning</p>', unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🏠 Rent Prediction", "📊 Market Insights", "💡 How It Works"])
    
    with tab1:
        # User Inputs
        furnishing_options = ("Furnished", "Semifurnished", "Unfurnished")
        available_for_options = ("Bachelors", "Family", "All")
        
        st.write("### 🏠 Property Details")
        
        furnishing = st.selectbox("🏡 Furnishing Status", furnishing_options)
        available_for = st.selectbox("👨‍👩‍👧 Availability", available_for_options)

        col1, col2, col3 = st.columns(3)

        with col1:
            rooms = st.number_input("🛏️ Number of Rooms", min_value=1, max_value=8, value=2)
        
        with col2:
            bathrooms = st.number_input("🚿 Number of Bathrooms", min_value=1, max_value=5, value=2)

        with col3:
            area = st.number_input("📏 Area in sqft", min_value=100, max_value=3000, value=1000, step=50)

        st.write("### 📌 Expected Rent Range")
        st.success("💰 Rent can vary between **₹5,000 to ₹2,50,000** per month depending on the area and amenities.")

        # Button to predict rent
        if st.button("🔍 Calculate Rent"):
            X = np.array([[rooms, bathrooms, area, furnishing, available_for]], dtype=object)
            X[:, 3] = le_furnishing.transform(X[:, 3])
            X[:, 4] = le_available_for.transform(X[:, 4])
            X = X.astype(float)

            rent = forest.predict(X)
            st.subheader(f"🏠 **The Estimated Rent is ₹{rent[0]:,.2f} per Month** 🎯")
            
            # Additional insights based on prediction
            st.write("### 📈 Rental Insights")
            if rent[0] < 10000:
                st.info("💡 This seems to be a budget-friendly property, suitable for students or young professionals.")
            elif rent[0] < 25000:
                st.info("💡 A mid-range property with good potential for rentals.")
            else:
                st.info("💡 A premium property with high-end amenities and location advantages.")

    with tab2:
        # Market Insights Visualization
        st.write("### 🏘️ Pune Rental Market Overview")
        
        # Sample data for visualization (replace with actual data if available)
        areas = ['Koregaon Park', 'Hinjewadi', 'Wakad', 'Baner', 'Kharadi']
        avg_rents = [35000, 25000, 20000, 22000, 30000]
        
        fig = px.bar(
            x=areas, 
            y=avg_rents, 
            title='Average Monthly Rent by Popular Pune Localities',
            labels={'x': 'Area', 'y': 'Average Rent (₹)'},
            color=avg_rents,
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig)
        
        st.write("### 🔍 Rental Trends")
        trends = {
            "2022": "Moderate Growth",
            "2023": "Steady Increase",
            "2024 (Projected)": "High Potential"
        }
        
        trend_df = pd.DataFrame.from_dict(trends, orient='index', columns=['Trend'])
        st.dataframe(trend_df)

    with tab3:
        st.write("### 💡 How Our Rent Prediction Works")
        st.write("""
        Our machine learning model uses a sophisticated Random Forest algorithm to estimate rental prices based on:
        
        1. 🛏️ Number of Rooms: More rooms typically mean higher rent
        2. 🚿 Number of Bathrooms: Additional bathrooms increase property value
        3. 📏 Area in Square Feet: Larger spaces command higher rents
        4. 🏡 Furnishing Status: Fully furnished properties are more expensive
        5. 👥 Availability: Different tenant preferences affect pricing
        """)
        
        st.info("🤖 The model is trained on extensive Pune real estate data to provide accurate predictions.")

    # Footer
    st.markdown(
        '<div class="footer">🔹 Made with ❤️ by Vedant Kale | Machine Learning Project</div>', 
        unsafe_allow_html=True
    )

# Run the application
def main():
    show_sidebar()
    show_predict_page()

if __name__ == "__main__":
    main()