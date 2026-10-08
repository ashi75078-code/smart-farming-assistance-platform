# 🌱 Smart Farming Assistance Platform

An AI-powered web application designed to help farmers make better decisions related to crop selection, plant disease detection, irrigation, and weather conditions.

## 📌 Project Overview

The **Smart Farming Assistance Platform** combines Artificial Intelligence, Machine Learning, Web Development, and Data Analysis to provide useful farming recommendations through a simple web interface.

The system takes agricultural and environmental inputs from the user and provides intelligent recommendations that can help improve farming decisions.

## 🎯 Objectives

- Help farmers select suitable crops based on soil and weather conditions.
- Detect plant diseases from uploaded leaf images.
- Provide irrigation recommendations based on environmental conditions.
- Provide weather information and farming-related weather advisories.
- Demonstrate the practical use of AI and Machine Learning in agriculture.
- Develop a simple and user-friendly platform for smart farming assistance.

## 🚀 Features

### 1. 🌾 Crop Recommendation AI

The Crop Recommendation module predicts a suitable crop based on:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

A Machine Learning model is trained using agricultural data and provides a crop recommendation based on the given conditions.

### 2. 🍃 Plant Disease Detection AI

Users can upload an image of a plant leaf.

The system uses a Deep Learning image classification model to analyze the image and predict the possible plant disease.

The result includes:

- Predicted disease/class
- Prediction confidence

### 3. 💧 Irrigation Advisor

The Irrigation Advisor provides a recommendation based on:

- Soil moisture
- Temperature
- Humidity
- Rainfall
- Crop water requirements

The system provides recommendations such as:

- Irrigation Recommended
- Light Irrigation Recommended
- No Irrigation Needed
- Monitor Soil Moisture

### 4. 🌦️ Weather Advisory

The Weather Advisory module provides current weather information and short-term forecasts.

It uses weather data to provide farming-related suggestions based on:

- Temperature
- Humidity
- Rainfall
- Rain probability
- Wind speed

## 🧠 AI & Machine Learning

The project uses different approaches for different farming tasks.

### Crop Recommendation

A **Random Forest Classifier** is used for crop prediction.

Input parameters:

```text
N
P
K
Temperature
Humidity
pH
Rainfall
