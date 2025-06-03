import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("C:\\Users\\Admin\\Desktop\\vs\\cleaned_tourdata.csv")
le_attraction = LabelEncoder()
data['AttractionType_en_encoded'] = le_attraction.fit_transform(data['AttractionType_en'])
le_visit_mode = LabelEncoder()
data['VisitModeId'] = le_visit_mode.fit_transform(data['VisitMode'])
st.sidebar.title("Tourism Analytics App")
st.sidebar.header("User Input")
continent = st.sidebar.selectbox("ContinentId", sorted(data["ContinentId"].unique()))
region = st.sidebar.selectbox("RegionId", sorted(data["RegionId"].unique()))
country = st.sidebar.selectbox("CountryId", sorted(data["CountryId"].unique()))
city = st.sidebar.selectbox("CityId", sorted(data["CityId"].unique()))
attraction_type = st.sidebar.selectbox("Attraction Type", sorted(data["AttractionType_en"].unique()))
visit_year = st.sidebar.slider("Visit Year", 2000, 2025, 2022)
visit_month = st.sidebar.slider("Visit Month", 1, 12, 6)
rating = st.sidebar.slider("Rating", 1, 5, 3)
attraction_type_encoded = le_attraction.transform([attraction_type])[0]
user_input = pd.DataFrame({
    'ContinentId': [continent],
    'RegionId': [region],
    'CountryId': [country],
    'CityId': [city],
    'AttractionType_en_encoded': [attraction_type_encoded],
    'VisitYear': [visit_year],
    'VisitMonth': [visit_month],
    'Rating': [rating]
})

x = data[['ContinentId', 'RegionId', 'CountryId', 'CityId',
          'AttractionType_en_encoded', 'VisitYear', 'VisitMonth', 'Rating']]
y = data['VisitModeId']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(random_state=42, max_depth=10, min_samples_leaf=1, min_samples_split=5)
classifier.fit(x_train, y_train)
visit_mode_pred_id = classifier.predict(user_input)[0]
visit_mode_pred = le_visit_mode.inverse_transform([visit_mode_pred_id])[0]
st.subheader("Predicted Visit Mode")
st.write(f"The predicted visit mode for the user is: **{visit_mode_pred}**")
cluster_features = x.copy()
cluster_features['VisitModeId'] = y
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_features['Group'] = kmeans.fit_predict(cluster_features)
user_input_cluster = user_input.copy()
user_input_cluster['VisitModeId'] = visit_mode_pred_id
user_group = kmeans.predict(user_input_cluster)[0]
data['Group'] = cluster_features['Group']
recommendations = data[data['Group'] == user_group][['CityId', 'AttractionType_en']].drop_duplicates().head(5)
st.subheader("Recommended Attractions")
st.table(recommendations)
