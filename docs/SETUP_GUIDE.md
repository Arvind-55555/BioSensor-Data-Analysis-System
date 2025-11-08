# BioSensor Data Analysis System - Complete Setup Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Frontend Setup (React Dashboard)](#frontend-setup)
3. [Backend Setup (Python ML Models)](#backend-setup)
4. [Project Structure](#project-structure)
5. [Running the Application](#running-the-application)
6. [Training Models](#training-models)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Software Requirements
- **Node.js**: v16.x or higher ([Download](https://nodejs.org/))
- **Python**: 3.8 or higher ([Download](https://www.python.org/))
- **Git**: Latest version ([Download](https://git-scm.com/))
- **npm** or **yarn**: Package manager (comes with Node.js)

### Verify Installation
```bash
node --version  # Should show v16.x or higher
python --version  # Should show 3.8 or higher
git --version
```

---

## 🎨 Frontend Setup (React Dashboard)

### Step 1: Create React Project

```bash
# Create new directory
mkdir biosensor-analysis
cd biosensor-analysis

# Initialize React app
npx create-react-app frontend
cd frontend
```

### Step 2: Install Dependencies

```bash
# Install required packages
npm install recharts lucide-react

# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### Step 3: Configure Tailwind CSS

**Update `tailwind.config.js`:**
```javascript
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**Update `src/index.css`:**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
```

### Step 4: Add Application Code

Replace `src/App.js` with the complete BioSensor application code (from the artifacts provided).

### Step 5: Update package.json

Add these scripts to your `package.json`:
```json
{
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  }
}
```

### Step 6: Test Frontend

```bash
npm start
```

Visit `http://localhost:3000` - you should see the BioSensor dashboard!

---

## 🐍 Backend Setup (Python ML Models)

### Step 1: Create Python Environment

```bash
# Go back to project root
cd ..

# Create models directory
mkdir models
cd models

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Python Dependencies

Create `requirements.txt` (content provided in artifacts):

```bash
pip install -r requirements.txt
```

### Step 3: Add Model Files

Add these three Python files to the `models/` directory:
1. `contamination_model.py`
2. `fermentation_model.py`
3. `growth_model.py`

(Complete code provided in artifacts above)

---

## 📁 Project Structure

After setup, your project should look like this:

```
biosensor-analysis/
│
├── frontend/                      # React Application
│   ├── node_modules/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js                # Main application
│   │   ├── index.js
│   │   ├── index.css
│   │   └── App.css
│   ├── package.json
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── models/                        # Python ML Models
│   ├── venv/                     # Virtual environment
│   ├── contamination_model.py
│   ├── fermentation_model.py
│   ├── growth_model.py
│   ├── requirements.txt
│   └── trained_models/           # Saved models (created after training)
│       ├── contamination_model.pkl
│       ├── fermentation_model.pkl
│       └── growth_model.pkl
│
├── data/                          # Sample data (optional)
│   └── sample_sensor_data.csv
│
├── docs/                          # Documentation
│   ├── API.md
│   ├── MODEL_DOCUMENTATION.md
│   └── DEPLOYMENT.md
│
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🚀 Running the Application

### Start Frontend Dashboard

```bash
cd frontend
npm start
```

The dashboard will open at `http://localhost:3000`

### Features Available:
- ✅ Real-time sensor monitoring
- ✅ Live data visualization
- ✅ Contamination risk prediction
- ✅ Fermentation success analysis
- ✅ Cell growth phase detection
- ✅ Alert system

---

## 🎓 Training Models

### Train Contamination Detection Model

```bash
cd models
source venv/bin/activate  # Activate virtual environment

python contamination_model.py
```

**Output:**
- Training metrics and performance report
- Confusion matrix visualization
- ROC curve
- Saved model: `contamination_model.pkl`

### Train Fermentation Success Model

```bash
python fermentation_model.py
```

**Output:**
- R² score and RMSE
- Feature importance plot
- Predictions vs actual plot
- Saved model: `fermentation_model.pkl`

### Train Cell Growth Model

```bash
python growth_model.py
```

**Output:**
- Phase classification report
- Growth rate prediction metrics
- Confusion matrix
- Feature importance plots
- Saved model: `growth_model.pkl`

### Using Trained Models

```python
from contamination_model import ContaminationDetectionModel
import pandas as pd

# Load trained model
model = ContaminationDetectionModel()
model.load_model('contamination_model.pkl')

# Prepare your data
sensor_data = pd.DataFrame({
    'pH': [7.0, 7.1, 7.2],
    'DO': [42, 41, 40],
    'CO2': [5.0, 5.2, 5.3],
    'temperature': [37, 37.1, 37.2]
})

# Create features
features = model.create_features(sensor_data)

# Predict
predictions = model.predict_with_risk_level(features)
print(predictions)
```

---

## 🌐 Deployment

### Frontend Deployment (Netlify/Vercel)

1. **Build the frontend:**
```bash
cd frontend
npm run build
```

2. **Deploy to Netlify:**
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=build
```

3. **Or deploy to Vercel:**
```bash
npm install -g vercel
vercel --prod
```

### Backend Deployment (Flask API - Optional)

Create `api/app.py`:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load models
contamination_model = joblib.load('contamination_model.pkl')
fermentation_model = joblib.load('fermentation_model.pkl')
growth_model = joblib.load('growth_model.pkl')

@app.route('/predict/contamination', methods=['POST'])
def predict_contamination():
    data = request.json
    df = pd.DataFrame([data])
    features = contamination_model['model'].create_features(df)
    prediction = contamination_model['model'].predict(features)
    return jsonify({'probability': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
```

Install Flask:
```bash
pip install flask flask-cors
```

Run API:
```bash
python app.py
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. **Module not found errors**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. **Tailwind CSS not working**
```bash
# Rebuild Tailwind
npx tailwindcss -i ./src/index.css -o ./dist/output.css --watch
```

#### 3. **Port already in use**
```bash
# Change port in package.json
"start": "PORT=3001 react-scripts start"
```

#### 4. **Python model training fails**
- Check Python version: `python --version` (need 3.8+)
- Ensure scikit-learn is installed: `pip install scikit-learn`
- Check available memory (models need ~2GB RAM)

#### 5. **React app won't start**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

---

## 📊 Testing the System

### Frontend Testing

```bash
cd frontend
npm test
```

### Model Testing

```python
# Test contamination model
python -m pytest tests/test_contamination_model.py

# Test fermentation model
python -m pytest tests/test_fermentation_model.py
```

---

## 📚 Additional Resources

- **React Documentation**: https://react.dev/
- **Recharts Documentation**: https://recharts.org/
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Tailwind CSS**: https://tailwindcss.com/docs

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Contact: arvind.saane.111@gmail.com

---

**Happy Coding! 🚀🔬**
