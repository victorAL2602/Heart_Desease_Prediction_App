from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import openai

app = Flask(__name__)

# Load and preprocess the data
file_path = 'heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(file_path)
feature_columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                   'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
target_column = 'DEATH_EVENT'
X = data[feature_columns]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def generate_recommendations(features):
    recommendations = []
    if features['high_blood_pressure'] == 1:
        recommendations.append("Consult with your doctor about managing your blood pressure.")
    if features['smoking'] == 1:
        recommendations.append("Consider quitting smoking. It significantly reduces the risk of heart disease.")
    if features['diabetes'] == 1:
        recommendations.append("Work with your healthcare provider to manage your blood sugar levels.")
    if features['serum_creatinine'] > 1.5:
        recommendations.append("Monitor your kidney function regularly.")
    if features['serum_sodium'] < 135:
        recommendations.append("Ensure you have a balanced diet to maintain proper sodium levels.")
    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    content = request.form.to_dict()
    content = {key: float(value) for key, value in content.items()}
    input_data = pd.DataFrame([content])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    recommendations = generate_recommendations(content)
    
    response = {
        'prediction': int(prediction),
        'recommendations': recommendations
    }
    return jsonify(response)

@app.route('/explain', methods=['POST'])
def explain():
    content = request.json
    explanation = f"Explanation for data: {content}"
    openai.api_key = 'YOUR_OPENAI_API_KEY'
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=f"Explain this prediction: {explanation}",
        max_tokens=50
    )
    return jsonify(response.choices[0].text.strip())

if __name__ == '__main__':
    app.run(debug=True)
