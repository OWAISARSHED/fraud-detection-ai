# AI Fraud Detection System

This project is a comprehensive AI-powered fraud detection system for financial transactions. It utilizes a FastAPI backend to provide real-time predictions using multiple machine learning and deep learning models. It also features an interactive frontend dashboard for monitoring and analyzing transactions.

## Features

- **Real-time Fraud Analysis**: Evaluates transactions instantly to determine fraud probability.
- **Ensemble Modeling**: Uses a combination of multiple models for robust predictions:
  - Isolation Forest
  - Random Forest
  - XGBoost
  - Autoencoder (Deep Learning)
- **Interactive Dashboard**: A frontend interface to view metrics, simulate transactions, and inspect risk levels.
- **RESTful API**: Exposes endpoints for integration with other services.

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Machine Learning**: scikit-learn, XGBoost, PyTorch/TensorFlow (for Autoencoder), Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript

## API Endpoints

- `GET /api/health`: Check the status of the API and loaded models.
- `POST /api/train`: Trigger model training in the background.
- `POST /api/predict`: Analyze a single transaction for fraud probability.
- `GET /api/predict/random`: Generate and analyze a random transaction.
- `GET /api/metrics`: Retrieve training metrics for all models.
- `GET /api/transactions`: Get recent transaction history.
- `GET /api/simulate`: Simulate a batch of transactions and return analysis.
- `GET /api/feature_importance`: Get feature importance from tree-based models.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd "Task 3_AI for Fraud Detection"
   ```

2. **Install dependencies:**
   Make sure you have Python installed. Then, run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Start the FastAPI server:
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:8000`.

## Directory Structure

- `app.py`: Main FastAPI application file.
- `train.py`: Script for training the models.
- `fraud_detection/`: Contains data generators and model definitions.
- `frontend/`: Contains the HTML, CSS, and JS for the web dashboard.
- `models/`: Stores the trained and serialized models (`.pkl` and `.json` files).
- `requirements.txt`: Python package dependencies.

## License

This project is licensed under the MIT License.
