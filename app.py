from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_from_directory
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

import os
import joblib
import requests
import torch

from PIL import Image
from transformers import AutoModelForImageClassification
from torchvision import transforms


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = "smart-farming-secret-key"

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# TEMPORARY USER DATABASE
# =========================================================

users = {}


# =========================================================
# CROP RECOMMENDATION MODEL
# =========================================================

CROP_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "crop_model.pkl"
)

try:

    crop_model = joblib.load(
        CROP_MODEL_PATH
    )

    print(
        "Crop Recommendation Model Loaded Successfully!"
    )

except Exception as e:

    crop_model = None

    print(
        "Crop Model Loading Error:",
        e
    )


# =========================================================
# PLANT DISEASE AI MODEL
# =========================================================

DISEASE_MODEL_NAME = (
    "mesabo/agri-plant-disease-resnet50"
)

try:

    print(
        "Loading Plant Disease AI model..."
    )

    disease_model = (
        AutoModelForImageClassification
        .from_pretrained(
            DISEASE_MODEL_NAME
        )
    )

    disease_model.eval()

    print(
        "Plant Disease AI Model Loaded Successfully!"
    )

    print(
        "Number of Disease Classes:",
        len(
            disease_model.config.id2label
        )
    )

except Exception as e:

    disease_model = None

    print(
        "Disease Model Loading Error:",
        e
    )


# =========================================================
# DISEASE IMAGE TRANSFORMATION
# =========================================================

disease_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# ALLOWED IMAGE TYPES
# =========================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        if not name or not email or not password:

            flash(
                "Please fill all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if email in users:

            flash(
                "An account with this email already exists.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        users[email] = {

            "name": name,

            "password": generate_password_hash(
                password
            )
        }

        flash(
            "Account created successfully. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = users.get(
            email
        )

        if (
            user
            and
            check_password_hash(
                user["password"],
                password
            )
        ):

            session["user_email"] = email

            session["user_name"] = user[
                "name"
            ]

            flash(
                "Login successful!",
                "success"
            )

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid email or password.",
            "error"
        )

    return render_template(
        "login.html"
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if "user_email" not in session:

        return redirect(
            url_for("login")
        )

    name = session.get(
        "user_name",
        "Farmer"
    )

    return render_template(
        "dashboard.html",
        name=name
    )


# =========================================================
# CROP RECOMMENDATION
# =========================================================

@app.route(
    "/crop-recommendation",
    methods=["GET", "POST"]
)
def crop_recommendation():

    if "user_email" not in session:

        return redirect(
            url_for("login")
        )

    recommendation = None

    error = None

    if request.method == "POST":

        try:

            nitrogen = float(
                request.form.get(
                    "nitrogen"
                )
            )

            phosphorus = float(
                request.form.get(
                    "phosphorus"
                )
            )

            potassium = float(
                request.form.get(
                    "potassium"
                )
            )

            temperature = float(
                request.form.get(
                    "temperature"
                )
            )

            humidity = float(
                request.form.get(
                    "humidity"
                )
            )

            ph = float(
                request.form.get(
                    "ph"
                )
            )

            rainfall = float(
                request.form.get(
                    "rainfall"
                )
            )

            features = [[

                nitrogen,

                phosphorus,

                potassium,

                temperature,

                humidity,

                ph,

                rainfall

            ]]

            if crop_model is None:

                error = (
                    "Crop recommendation model "
                    "is not available."
                )

            else:

                prediction = (
                    crop_model.predict(
                        features
                    )
                )

                recommendation = prediction[0]

        except Exception as e:

            print(
                "Crop Recommendation Error:",
                e
            )

            error = (
                "Please enter valid numeric values."
            )

    return render_template(

        "crop_recommendation.html",

        recommendation=recommendation,

        error=error
    )


# =========================================================
# PLANT DISEASE DETECTION
# =========================================================

@app.route(
    "/disease-detection",
    methods=["GET", "POST"]
)
def disease_detection():

    if "user_email" not in session:

        return redirect(
            url_for("login")
        )

    prediction = None

    confidence = None

    image_url = None

    error = None

    if request.method == "POST":

        try:

            if (
                "leaf_image"
                not in request.files
            ):

                error = (
                    "Please select an image."
                )

            else:

                file = request.files[
                    "leaf_image"
                ]

                if file.filename == "":

                    error = (
                        "Please select an image."
                    )

                elif not allowed_file(
                    file.filename
                ):

                    error = (
                        "Only JPG, JPEG, PNG "
                        "and WEBP images are allowed."
                    )

                elif disease_model is None:

                    error = (
                        "Disease detection model "
                        "is not available."
                    )

                else:

                    filename = file.filename

                    filepath = os.path.join(

                        app.config[
                            "UPLOAD_FOLDER"
                        ],

                        filename
                    )

                    file.save(
                        filepath
                    )

                    image = Image.open(
                        filepath
                    ).convert(
                        "RGB"
                    )

                    image_tensor = (
                        disease_transform(
                            image
                        )
                        .unsqueeze(0)
                    )

                    with torch.no_grad():

                        outputs = (
                            disease_model(
                                pixel_values=
                                image_tensor
                            )
                        )

                        probabilities = (
                            torch.softmax(
                                outputs.logits,
                                dim=1
                            )
                        )

                        confidence_value, predicted_class = (
                            torch.max(
                                probabilities,
                                dim=1
                            )
                        )

                    predicted_index = (
                        predicted_class
                        .item()
                    )

                    confidence = round(

                        confidence_value
                        .item()
                        * 100,

                        2
                    )

                    prediction = (
                        disease_model
                        .config
                        .id2label
                        .get(
                            predicted_index,
                            "Unknown Disease"
                        )
                    )

                    image_url = url_for(

                        "uploaded_file",

                        filename=filename
                    )

        except Exception as e:

            print(
                "Disease Detection Error:",
                e
            )

            error = (
                "Unable to process the image."
            )

    return render_template(

        "disease_detection.html",

        prediction=prediction,

        confidence=confidence,

        image_url=image_url,

        error=error
    )


# =========================================================
# UPLOADED IMAGE ROUTE
# =========================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename
    )


# =========================================================
# IRRIGATION ADVISOR
# =========================================================

@app.route(
    "/irrigation-advisor",
    methods=["GET", "POST"]
)
def irrigation_advisor():

    if "user_email" not in session:

        return redirect(
            url_for("login")
        )

    result = None

    score = None

    error = None

    if request.method == "POST":

        try:

            crop = request.form.get(
                "crop",
                ""
            ).strip().lower()

            soil_type = request.form.get(
                "soil_type",
                ""
            ).strip().lower()

            soil_moisture = float(
                request.form.get(
                    "soil_moisture"
                )
            )

            temperature = float(
                request.form.get(
                    "temperature"
                )
            )

            humidity = float(
                request.form.get(
                    "humidity"
                )
            )

            rainfall = float(
                request.form.get(
                    "rainfall"
                )
            )

            irrigation_score = 0

            # Soil moisture rules

            if soil_moisture < 25:

                irrigation_score += 4

            elif soil_moisture < 40:

                irrigation_score += 2

            elif soil_moisture < 55:

                irrigation_score += 1

            else:

                irrigation_score -= 3

            # Temperature rules

            if temperature >= 35:

                irrigation_score += 2

            elif temperature >= 28:

                irrigation_score += 1

            elif temperature < 18:

                irrigation_score -= 1

            # Humidity rules

            if humidity < 40:

                irrigation_score += 2

            elif humidity < 60:

                irrigation_score += 1

            elif humidity > 80:

                irrigation_score -= 2

            # Rainfall rules

            if rainfall >= 50:

                irrigation_score -= 4

            elif rainfall >= 25:

                irrigation_score -= 2

            elif rainfall >= 10:

                irrigation_score -= 1

            # Crop water requirement

            high_water_crops = [

                "rice",

                "sugarcane",

                "banana",

                "cotton"

            ]

            low_water_crops = [

                "chickpea",

                "kidneybeans",

                "lentil",

                "blackgram"

            ]

            if crop in high_water_crops:

                irrigation_score += 1

            elif crop in low_water_crops:

                irrigation_score -= 1

            # Final recommendation

            if irrigation_score >= 5:

                result = (
                    "Irrigation Recommended 💧"
                )

            elif irrigation_score >= 2:

                result = (
                    "Light Irrigation Recommended 🌱"
                )

            elif irrigation_score <= -2:

                result = (
                    "No Irrigation Needed ✅"
                )

            else:

                result = (
                    "Monitor Soil Moisture 👀"
                )

            score = irrigation_score

        except Exception as e:

            print(
                "Irrigation Error:",
                e
            )

            error = (
                "Please enter valid values."
            )

    return render_template(

        "irrigation_advisor.html",

        result=result,

        score=score,

        error=error
    )


# =========================================================
# WEATHER ADVISORY
# =========================================================

@app.route(
    "/weather-advisory",
    methods=["GET", "POST"]
)
def weather_advisory():

    if "user_email" not in session:

        return redirect(
            url_for("login")
        )

    weather = None

    advisory = None

    error = None

    if request.method == "POST":

        try:

            city = request.form.get(
                "city",
                ""
            ).strip()

            if not city:

                error = (
                    "Please enter a city or location."
                )

            else:

                # -------------------------------------------------
                # STEP 1: FIND LOCATION
                # -------------------------------------------------

                geo_url = (
                    "https://geocoding-api.open-meteo.com/"
                    "v1/search"
                )

                geo_response = requests.get(

                    geo_url,

                    params={

                        "name": city,

                        "count": 1,

                        "language": "en",

                        "format": "json"
                    },

                    timeout=10
                )

                geo_data = (
                    geo_response.json()
                )

                if not geo_data.get(
                    "results"
                ):

                    error = (
                        "Location not found. "
                        "Please enter a valid city."
                    )

                else:

                    location = (
                        geo_data[
                            "results"
                        ][0]
                    )

                    latitude = (
                        location[
                            "latitude"
                        ]
                    )

                    longitude = (
                        location[
                            "longitude"
                        ]
                    )

                    location_name = (
                        location[
                            "name"
                        ]
                    )

                    country = location.get(
                        "country",
                        ""
                    )

                    # -------------------------------------------------
                    # STEP 2: GET WEATHER
                    # -------------------------------------------------

                    weather_url = (
                        "https://api.open-meteo.com/"
                        "v1/forecast"
                    )

                    weather_response = (
                        requests.get(

                            weather_url,

                            params={

                                "latitude":
                                    latitude,

                                "longitude":
                                    longitude,

                                "current": (
                                    "temperature_2m,"
                                    "relative_humidity_2m,"
                                    "precipitation,"
                                    "weather_code,"
                                    "wind_speed_10m"
                                ),

                                "daily": (
                                    "temperature_2m_max,"
                                    "temperature_2m_min,"
                                    "precipitation_probability_max,"
                                    "precipitation_sum"
                                ),

                                "forecast_days": 3,

                                "timezone": "auto"
                            },

                            timeout=10
                        )
                    )

                    weather_data = (
                        weather_response.json()
                    )

                    current = (
                        weather_data.get(
                            "current",
                            {}
                        )
                    )

                    daily = (
                        weather_data.get(
                            "daily",
                            {}
                        )
                    )

                    temperature = (
                        current.get(
                            "temperature_2m"
                        )
                    )

                    humidity = (
                        current.get(
                            "relative_humidity_2m"
                        )
                    )

                    precipitation = (
                        current.get(
                            "precipitation"
                        )
                    )

                    wind_speed = (
                        current.get(
                            "wind_speed_10m"
                        )
                    )

                    weather_code = (
                        current.get(
                            "weather_code"
                        )
                    )

                    rain_probability = (

                        daily.get(

                            "precipitation_probability_max",

                            [0]

                        )[0]
                    )

                    rain_sum = (

                        daily.get(

                            "precipitation_sum",

                            [0]

                        )[0]
                    )

                    # -------------------------------------------------
                    # WEATHER DESCRIPTION
                    # -------------------------------------------------

                    weather_descriptions = {

                        0:
                            "Clear Sky",

                        1:
                            "Mainly Clear",

                        2:
                            "Partly Cloudy",

                        3:
                            "Overcast",

                        45:
                            "Fog",

                        48:
                            "Fog",

                        51:
                            "Light Drizzle",

                        53:
                            "Moderate Drizzle",

                        55:
                            "Heavy Drizzle",

                        61:
                            "Light Rain",

                        63:
                            "Moderate Rain",

                        65:
                            "Heavy Rain",

                        71:
                            "Light Snow",

                        73:
                            "Moderate Snow",

                        75:
                            "Heavy Snow",

                        80:
                            "Rain Showers",

                        81:
                            "Moderate Rain Showers",

                        82:
                            "Heavy Rain Showers",

                        95:
                            "Thunderstorm",

                        96:
                            "Thunderstorm with Hail",

                        99:
                            "Thunderstorm with Heavy Hail"
                    }

                    condition = (
                        weather_descriptions.get(

                            weather_code,

                            "Variable Weather"
                        )
                    )

                    # -------------------------------------------------
                    # FARMING ADVISORY
                    # -------------------------------------------------

                    if rain_probability >= 70:

                        advisory = (
                            "Rain is likely. Avoid "
                            "unnecessary irrigation and "
                            "consider delaying field "
                            "spraying activities."
                        )

                    elif temperature >= 35:

                        advisory = (
                            "High temperature detected. "
                            "Monitor soil moisture closely "
                            "and provide irrigation if "
                            "required."
                        )

                    elif humidity >= 80:

                        advisory = (
                            "High humidity detected. "
                            "Monitor crops for fungal "
                            "diseases and avoid excessive "
                            "irrigation."
                        )

                    elif temperature <= 10:

                        advisory = (
                            "Low temperature detected. "
                            "Protect sensitive crops from "
                            "possible cold stress."
                        )

                    elif wind_speed >= 30:

                        advisory = (
                            "Strong wind conditions detected. "
                            "Avoid spraying pesticides or "
                            "fertilizers during strong winds."
                        )

                    else:

                        advisory = (
                            "Weather conditions are relatively "
                            "favorable. Continue normal crop "
                            "monitoring and irrigation practices."
                        )

                    # -------------------------------------------------
                    # STORE WEATHER DATA
                    # -------------------------------------------------

                    weather = {

                        "city":
                            location_name,

                        "country":
                            country,

                        "temperature":
                            temperature,

                        "humidity":
                            humidity,

                        "precipitation":
                            precipitation,

                        "wind_speed":
                            wind_speed,

                        "condition":
                            condition,

                        "rain_probability":
                            rain_probability,

                        "rain_sum":
                            rain_sum
                    }

        except Exception as e:

            print(
                "Weather Error:",
                e
            )

            error = (
                "Unable to retrieve weather "
                "information. Please try again."
            )

    return render_template(

        "weather_advisory.html",

        weather=weather,

        advisory=advisory,

        error=error
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("home")
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        " Smart Farming Assistance Platform"
    )

    print(
        "======================================"
    )

    print(
        "Server starting..."
    )

    app.run(
        debug=True
    )