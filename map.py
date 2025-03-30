import streamlit as st
import folium
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

#Load pre-trained model and other resources 
model = joblib.load('ensemble_model.pkl') 
label_encoder = joblib.load('label_encoder.pkl')  
vectorizer = joblib.load('vectorizer.pkl')  

#Load dataset 
# df = pd.read_csv("Updated_Competition_Dataset.csv")
# uncomment this to run the code 

#Remove timestamp column if present
if 'Date' in df.columns:
    df = df.drop(columns=['Date'])

#Helper function to extract text from the PDF using pdfplumber
def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

#Helper function: Assign colors to crimes
def get_color(category):
    color_map = {
        'ROBBERY': 'red', 'BURGLARY': 'blue', 'VEHICLE THEFT': 'green',
        'LARCENY/THEFT': 'orange', 'ARSON': 'purple', 'FRAUD': 'pink',
        'SUSPICIOUS OCC': 'yellow', 'VANDALISM': 'brown', 'DRUG/NARCOTIC': 'darkred',
        'WEAPON LAWS': 'darkgreen', 'KIDNAPPING': 'darkblue', 'OTHER OFFENSES': 'lightblue',
        'DISORDERLY CONDUCT': 'gray'
    }
    return color_map.get(category, 'gray')

#Helper function: Assign severity
severity_map = {
    'ROBBERY': 'High', 'BURGLARY': 'Medium', 'VEHICLE THEFT': 'Low',
    'LARCENY/THEFT': 'Medium', 'ARSON': 'High', 'FRAUD': 'Low',
    'SUSPICIOUS OCC': 'Medium', 'VANDALISM': 'Low', 'DRUG/NARCOTIC': 'High',
    'WEAPON LAWS': 'High', 'KIDNAPPING': 'High', 'OTHER OFFENSES': 'Low',
    'DISORDERLY CONDUCT': 'Low'
}

# Function to predict the crime category and severity based on description and location
def predict_crime(description, distance_from_center):
    description_vectorized = vectorizer.transform([description]).toarray()
    input_data = np.hstack((description_vectorized, np.array([[distance_from_center]])))
    
    predicted_category = model.predict(input_data)[0]
    predicted_category_name = label_encoder.inverse_transform([predicted_category])[0]
    
    # Map to severity
    severity = severity_map.get(predicted_category_name, 'Unknown')
    
    return predicted_category_name, severity

#UI Design 
st.set_page_config(page_title="Crime Dashboard", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            font-size: 18px;
            border-radius: 5px;
            padding: 12px 25px;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        .section-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin: 20px 0;
        }
        .section-container .stButton {
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("🏙️ CityX: Crime Analytics & Prediction Dashboard")

# Crime Data Analysis Section
with st.container():
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.subheader(" 📊 Crime Data Analysis")

    # Crime Category Distribution
    st.markdown("### Crime Category Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))  
    sns.set_theme(style="whitegrid")
    sns.countplot(y=df['Category'], order=df['Category'].value_counts().index, palette="Set2", ax=ax)
    ax.set_xlabel("Number of Crimes", fontsize=14)
    ax.set_ylabel("Crime Category", fontsize=14)
    ax.set_title("Crime Category Distribution", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    st.pyplot(fig)
    
    # Crime Severity Distribution 
    st.markdown("###  Crime Severity Levels")
    df['Severity'] = df['Category'].map(severity_map)
    severity_counts = df['Severity'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%',
           colors=['#e74c3c', '#f39c12', '#27ae60'], startangle=140, wedgeprops={'edgecolor': 'black'})
    ax.set_title("Crime Severity Distribution", fontsize=16)
    st.pyplot(fig)

    #  Crime Distance from Center 
    st.markdown("###  Crime Distance from Center")
    category_distance = df.groupby("Category")["distance_from_center"].agg(["mean"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=category_distance, x=category_distance.index, y="mean", palette="magma", ax=ax)
    ax.set_xlabel("Crime Category", fontsize=14)
    ax.set_ylabel("Mean Distance (km)", fontsize=14)
    ax.set_title("Average Distance from Center per Crime Category", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Incidents per Day of the Week 
with st.container():
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.subheader(" Number of Incidents per Day of the Week")

    # Calculate the number of incidents per day of the week
    day_counts = df['DayOfWeek'].value_counts().sort_index()

    # Plot the bar chart for incidents per day of the week
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=day_counts.index, y=day_counts.values, palette='viridis', ax=ax)
    ax.set_title('Number of Incidents per Day of the Week', fontsize=16)
    ax.set_xlabel('Day of the Week (0=Monday, 6=Sunday)', fontsize=14)
    ax.set_ylabel('Number of Incidents', fontsize=14)
    ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=45)

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)

#Crime Prediction Section 
with st.container():
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.subheader("📈 Crime Prediction")

    # Step 1: File Upload - Upload Police Report PDF
    uploaded_file = st.file_uploader("Upload Police Report PDF", type=["pdf"])

    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        pdf_text = extract_pdf_text(uploaded_file)
        st.text_area("Extracted Text", pdf_text, height=300)

        # Step 2: Data Extraction - This simulates extracted fields from the PDF
        st.subheader("Enter the crime details")

        # Simulated fields from the PDF extraction - Customize based on your needs
        description = st.text_input("Crime Description")

        # Correct way to set the minimum value for distance_from_center
        distance_from_center = st.number_input("Distance from Center (meters)")

        # Step 3: Crime Prediction - Using the trained model
        if st.button("Predict Crime Category and Severity"):
            if description:
                category, severity = predict_crime(description, distance_from_center)
                st.success(f"Predicted Crime Category: {category}")
                st.warning(f"Predicted Crime Severity: {severity}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Interactive Map 
with st.container():
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.subheader("📍 Crime Locations on Map")

    # Initialize a Fixed Map
    map_center = [df['Longitude (X)'].mean(), df['Latitude (Y)'].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=12)

    # Limit the number of markers to 500 for performance
    df_limited = df.head(500)

    # Add markers for each crime location with color based on the crime category
    for index, row in df_limited.iterrows():
        crime_category = row['Category']
        latitude = row['Longitude (X)']  # Swapped: Longitude is used for latitude
        longitude = row['Latitude (Y)']  # Swapped: Latitude is used for longitude
        
        folium.Marker(
            location=[latitude, longitude],
            popup=f"Category: {crime_category}",
            icon=folium.Icon(color=get_color(crime_category), icon='info-sign')
        ).add_to(crime_map)

    # Show the map
    st_folium(crime_map, width=700, height=500)
    
    st.markdown("</div>", unsafe_allow_html=True)
