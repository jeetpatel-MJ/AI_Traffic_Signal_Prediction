import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import json
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
import base64

# This function is included in case the model files don't exist.
# It performs the full training pipeline to create the necessary assets.
def train_model():
    file_path = 'smart_traffic_management_dataset.csv'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    df = pd.read_csv(file_path)

    # Feature Engineering: Extract hour from timestamp
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df.drop(columns=['timestamp'], inplace=True)

    # Encode categorical columns (one-hot encoding)
    categorical_cols = ['location_id', 'weather_condition', 'signal_status']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Normalize numerical features
    numerical_cols = ['traffic_volume', 'avg_vehicle_speed', 'vehicle_count_cars', 
                      'vehicle_count_trucks', 'vehicle_count_bikes', 'temperature', 'humidity', 'hour']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Separate features and target
    X = df.drop('accident_reported', axis=1)
    y = df['accident_reported']

    # Save feature columns for prediction
    feature_columns = X.columns.tolist()
    with open('plots/feature_columns.json', 'w') as f:
        json.dump(feature_columns, f, indent=4)

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Train Random Forest Model with Hyperparameter Tuning
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_balanced, y_train_balanced)
    clf = grid_search.best_estimator_

    # Save the model and scaler
    os.makedirs('plots', exist_ok=True)
    joblib.dump(clf, 'plots/model.joblib')
    joblib.dump(scaler, 'plots/scaler.joblib')

    # Save best parameters
    with open('plots/best_params.json', 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)

    # Predict & Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Visual Outputs
    os.makedirs('plots', exist_ok=True)

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

    # Feature Importance Bar Graph
    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns).nlargest(10)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feat_importances, y=feat_importances.index, color='green')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.savefig('plots/feature_importance.png')
    plt.close()

    # Feature Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    sns.heatmap(corr, cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

    # Chart.js Bar Chart for Performance Metrics
    chartjs_data = {
        "type": "bar",
        "data": {
            "labels": ["Accuracy", "Precision (0)", "Recall (0)", "F1-Score (0)", 
                       "Precision (1)", "Recall (1)", "F1-Score (1)"],
            "datasets": [{
                "label": "Model Performance Metrics",
                "data": [
                    accuracy,
                    class_report['0']['precision'],
                    class_report['0']['recall'],
                    class_report['0']['f1-score'],
                    class_report['1']['precision'],
                    class_report['1']['recall'],
                    class_report['1']['f1-score']
                ],
                "backgroundColor": ["#2ecc71", "#3498db", "#3498db", "#3498db", 
                                    "#e74c3c", "#e74c3c", "#e74c3c"],
                "borderColor": ["#27ae60", "#2980b9", "#2980b9", "#2980b9", 
                                 "#c0392b", "#c0392b", "#c0392b"],
                "borderWidth": 1
            }]
        },
        "options": {
            "responsive": True,
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {"display": True, "text": "Score"},
                    "max": 1
                },
                "x": {
                    "title": {"display": True, "text": "Metric"}
                }
            },
            "plugins": {
                "legend": {"display": True},
                "title": {"display": True, "text": f"Model Performance Metrics (Accuracy: {accuracy:.2%})"}
            }
        }
    }

    # Save Chart.js configuration
    with open('plots/performance_metrics_chartjs.json', 'w') as f:
        json.dump(chartjs_data, f, indent=4)
    print("\nChart.js configuration saved to 'plots/performance_metrics_chartjs.json'.")


# Check if model files exist, train if not
model_path = 'plots/model.joblib'
scaler_path = 'plots/scaler.joblib'
features_path = 'plots/feature_columns.json'

if not all(os.path.exists(path) for path in [model_path, scaler_path, features_path]):
    print("Model files not found. Training the model now...")
    train_model()

# Load the trained model, scaler, and feature columns
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        feature_columns = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Model, scaler, or feature columns file not found even after training. Ensure the dataset exists and training completes successfully.")

app = Flask(__name__, static_url_path='/plots', static_folder='plots')
app.secret_key = os.environ.get('FLASK_SECRET', 'dev_secret_for_local')

# Simple user store (file-based). Contains usernames, hashed passwords and roles.
USERS_FILE = os.path.join('plots', 'users.json')

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def ensure_default_users():
    users = load_users()
    changed = False
    if 'admin' not in users:
        users['admin'] = {
            'password': generate_password_hash('adminpass'),
            'role': 'admin'
        }
        changed = True
    if 'user' not in users:
        users['user'] = {
            'password': generate_password_hash('userpass'),
            'role': 'user'
        }
        changed = True
    if changed:
        save_users(users)

def reload_model():
    global model, scaler, feature_columns
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        feature_columns = json.load(f)

def role_required(role):
    def decorator(f):
        def wrapped(*args, **kwargs):
            if not session.get('user'):
                return redirect(url_for('login'))
            if session.get('role') != role:
                return "Forbidden", 403
            return f(*args, **kwargs)
        wrapped.__name__ = f.__name__
        return wrapped
    return decorator

ensure_default_users()

# Static mapping for location_id -> descriptive label (used if CSV unavailable)
LOCATION_MAP = {
    '1': 'Delhi, Connaught Place',
    '2': 'Mumbai, Andheri',
    '3': 'Bengaluru, Whitefield',
    '4': 'Ahmedabad, Gota',
    '5': 'Chennai, T. Nagar',
    '6': 'Kolkata, Salt Lake',
    '7': 'Hyderabad, Banjara Hills',
    '8': 'Pune, Koregaon Park',
    '9': 'Surat, Varachha',
    '10': 'Jaipur, MI Road',
    '11': 'Lucknow, Hazratganj',
    '12': 'Kanpur, Swaroop Nagar',
    '13': 'Nagpur, Sitabuldi',
    '14': 'Indore, Rajwada',
    '15': 'Bhopal, New Market'
}

def get_locations_from_csv():
    # Try to read unique location ids from CSV; fall back to LOCATION_MAP
    csv_path = 'smart_traffic_management_dataset.csv'
    try:
        import pandas as _pd
        df = _pd.read_csv(csv_path)
        if 'location_id' in df.columns:
            vals = df['location_id'].unique().tolist()
            out = []
            for v in vals:
                sv = str(v)
                label = LOCATION_MAP.get(sv, sv)
                out.append({'id': sv, 'label': label})
            return out
    except Exception:
        pass
    # default mapping
    return [{'id': k, 'label': v} for k, v in LOCATION_MAP.items()]


@app.route('/api/locations')
def api_locations():
    return jsonify(get_locations_from_csv())

# Ensure a small favicon exists in the plots folder so browsers don't 404
os.makedirs('plots', exist_ok=True)
favicon_path = os.path.join('plots', 'favicon.png')
if not os.path.exists(favicon_path):
    # 1x1 transparent PNG (very small) to use as favicon
    favicon_base64 = (
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='
    )
    with open(favicon_path, 'wb') as f:
        f.write(base64.b64decode(favicon_base64))


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('plots', 'favicon.png')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()
        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            session['role'] = user.get('role', 'user')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required')
            return render_template('register.html')
        users = load_users()
        if username in users:
            flash('Username already exists')
            return render_template('register.html')
        users[username] = {
            'password': generate_password_hash(password),
            'role': 'user'
        }
        save_users(users)
        flash('Account created — please sign in')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')


@app.route('/admin')
def admin():
    if not session.get('user') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    users = load_users()
    return render_template('admin.html', users=users)


@app.route('/admin/retrain', methods=['POST'])
def admin_retrain():
    if not session.get('user') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    # retrain (may take time)
    train_model()
    reload_model()
    flash('Model retrained and reloaded')
    return redirect(url_for('admin'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Require login
        if not session.get('user'):
            return jsonify({'error': 'Authentication required'}), 401
        # Get JSON data from request
        data = request.get_json()
        
        # Create DataFrame from input data
        input_data = pd.DataFrame([data])
        
        # Feature Engineering: Extract hour from timestamp
        if 'timestamp' in input_data.columns:
            input_data['hour'] = pd.to_datetime(input_data['timestamp']).dt.hour
            input_data.drop(columns=['timestamp'], inplace=True)
        
        # Encode categorical columns (one-hot encoding)
        categorical_cols = ['location_id', 'weather_condition', 'signal_status']
        for col in categorical_cols:
            if col in input_data.columns:
                unique_values = input_data[col].unique()
                for value in unique_values:
                    input_data[f"{col}_{value}"] = (input_data[col] == value).astype(int)
                input_data.drop(columns=[col], inplace=True)
        
        # Normalize numerical features
        numerical_cols = ['traffic_volume', 'avg_vehicle_speed', 'vehicle_count_cars', 
                          'vehicle_count_trucks', 'vehicle_count_bikes', 'temperature', 'humidity', 'hour']
        
        # Make a copy before scaling to avoid a warning
        numerical_data_to_scale = input_data[numerical_cols].copy()
        input_data[numerical_cols] = scaler.transform(numerical_data_to_scale)
        
        # Ensure input_data has the same columns as training data
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0].tolist()
        
        # Log prediction to file with user
        try:
            log_path = os.path.join('plots', 'predictions.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as lf:
                lf.write(json.dumps({
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'user': session.get('user'),
                    'input': data,
                    'prediction': int(prediction[0]),
                    'probability': probability
                }) + "\n")
        except Exception:
            pass

        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': probability
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/plots/<path:filename>')
def plots_files(filename):
    return send_from_directory('plots', filename)

if __name__ == '__main__':
    app.run(debug=True)
