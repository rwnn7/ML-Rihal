# ML-Rihal
# Crime Analytics and Prediction Dashboard

## Project Overview

The **Crime Analytics and Prediction Dashboard** is a web-based application developed using **Streamlit** for the front-end, and **Machine Learning** techniques to analyze and predict crime data. The dashboard provides an interactive platform to visualize crime statistics and make predictions about crime categories and severity based on the description of incidents.

### Key Features:
- **Crime Data Analysis**: Visualizes crime data including crime categories, severity, and the average distance of crimes from the city center.
- **Crime Prediction**: Predicts the category and severity of a crime based on its description and the distance from the city center using a pre-trained machine learning model.
- **Interactive Crime Locations Map**: Displays crime locations on an interactive map with color-coded markers, helping to identify patterns in crime distribution across the city.

This project integrates various libraries such as **Folium**, **Streamlit**, and **Seaborn** to enable rich visualizations and enhance user experience. The model was trained using historical crime data, which has been processed and is used for both prediction and analysis.

### How It Works:
1. **Crime Data Analysis**: 
   - The dashboard loads historical crime data to create interactive visualizations such as crime category distribution and crime severity levels.
   - It shows how crimes are distributed over different days of the week and calculates the distance of incidents from a central reference point.

2. **Crime Prediction**:
   - Users can input a crime description and distance from the center, and the app uses a pre-trained model to predict the crime category and severity (e.g., High, Medium, Low).

3. **Crime Map**:
   - A dynamic map is displayed to show the locations of crimes using markers. The map is color-coded by crime category, allowing users to visualize the spread of crime incidents over time.

## Project Setup

To run the app, you'll need to follow a few simple steps:

### 1. Download Necessary Resources:
Before running the app, download the following necessary resources:

- **ensemble_model.pkl**: The pre-trained machine learning model for crime prediction.
- **label_encoder.pkl**: The label encoder for converting categorical labels into numerical format.
- **vectorizer.pkl**: The TF-IDF vectorizer used to transform the text descriptions of crimes into numerical data.
- **Updated_Competition_Dataset.csv**: The dataset containing historical crime data for analysis.

You can download these files from the links provided in the project documentation.

### 2. Project Structure:
Ensure that the following files are placed in the root directory of your project:

- `app.py`: The main application file containing all the code for crime data analysis, prediction, and visualization.
- `ensemble_model.pkl`: Pre-trained machine learning model for crime prediction.
- `label_encoder.pkl`: Label encoder used to decode the model's predicted labels.
- `vectorizer.pkl`: TF-IDF vectorizer to process crime descriptions.
- `Updated_Competition_Dataset.csv`: The dataset with crime-related data.

### 3. Install Requirements:
The project requires several Python packages to run. You can install them by running the following command:

```bash
pip install -r requirements.txt
