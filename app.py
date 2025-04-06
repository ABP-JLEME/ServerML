from flask import Flask, request, jsonify
import pickle
import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import mean_squared_error
import numpy as np
import time

app = Flask(__name__)

# Ensure directory exists
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Load model based on user_id and item_id
def load_model(user_id, item_id):
    model_path = f'./models/{user_id}/{item_id}.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {user_id}/{item_id} not found.")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Save model
def save_model(user_id, item_id, model):
    directory = f"./models/{user_id}"
    ensure_directory(directory)
    filename = f"{directory}/{item_id}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Check if model exists
def is_model_exist(user_id, item_id):
    filename = f"./models/{user_id}/{item_id}.pkl"
    return os.path.exists(filename)

# Train XGBoost model
def train_xgboost(user_id, item_id, df):
    df = df.sort_values(by='tanggal')
    df['day'] = (df['tanggal'] - df['tanggal'].min()).dt.days
    df['day_of_week'] = df['tanggal'].dt.dayofweek
    df['month'] = df['tanggal'].dt.month
    df['moving_avg'] = df['jumlah_terjual'].rolling(window=3).mean().bfill()

    X = df[['day', 'day_of_week', 'month', 'moving_avg']]
    y = df['jumlah_terjual']
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5)
    start_time = time.time()
    model.fit(X, y)
    train_time = time.time() - start_time

    # Sliding window validation (90% train, 10% val)
    train_size = int(len(df) * 0.9)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Validation RMSE: {val_rmse}")
    
    save_model(user_id, item_id, model)

    return val_rmse, train_time

# Update existing model
def update_model(user_id, item_id, updated_data):
    model = load_model(user_id, item_id)
    updated_data['day'] = (updated_data['tanggal'] - updated_data['tanggal'].min()).dt.days
    updated_data['day_of_week'] = updated_data['tanggal'].dt.dayofweek
    updated_data['month'] = updated_data['tanggal'].dt.month
    updated_data['moving_avg'] = updated_data['jumlah_terjual'].rolling(window=3).mean().bfill()

    X = updated_data[['day', 'day_of_week', 'month', 'moving_avg']]
    y = updated_data['jumlah_terjual']

    start_time = time.time()
    model.fit(X, y, xgb_model=model)
    train_time = time.time() - start_time

    train_size = int(len(updated_data) * 0.9)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Validation RMSE: {val_rmse}")
    
    save_model(user_id, item_id, model)

    return val_rmse, train_time

# Predict future sales
def predict_sales(user_id, item_id, df, future_days=3):
    model = load_model(user_id, item_id)
    
    df = df.sort_values(by='tanggal')  
    df['day'] = (df['tanggal'] - df['tanggal'].min()).dt.days  
    df['day_of_week'] = df['tanggal'].dt.dayofweek
    df['month'] = df['tanggal'].dt.month
    df['moving_avg'] = df['jumlah_terjual'].rolling(window=3).mean().bfill()

    latest_day = df['day'].max()
    latest_date = df['tanggal'].max()

    future_dates = [latest_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
    
    future_df = pd.DataFrame({
        'day': [latest_day + i for i in range(1, future_days + 1)],
        'day_of_week': [date.dayofweek for date in future_dates],
        'month': [date.month for date in future_dates]
    })

    future_df['moving_avg'] = df['moving_avg'].iloc[-1] if not df['moving_avg'].empty else 0
    predictions = model.predict(future_df)
    return predictions.tolist()

@app.route('/train', methods=['POST'])
def train():
    try:
        param = request.get_json()
        user_id = param.get('user_id')
        item_id = param.get('item_id')
        
        data = pd.read_csv('./data/dummy.csv', parse_dates=['tanggal'])  # Ambil data dari server Firebase
        grouped_data = { (user, item): df.assign(tanggal=pd.to_datetime(df['tanggal'])) for (user, item), df in data.groupby(['user_id', 'item_id']) }
        
        if not is_model_exist(user_id, item_id):
            action = "trained"
            val_rmse, train_time = train_xgboost(user_id, item_id, grouped_data[(user_id, item_id)])
        else:
            action = "updated"
            val_rmse, train_time = update_model(user_id, item_id, grouped_data[(user_id, item_id)])
        
        return jsonify({
            "status": "success",
            "message": f"Model has been {action} successfully",
            "user_id": user_id,
            "item_id": item_id,
            "action": action,
            "training_time": f"{train_time:.2f} seconds",
            "model_accuracy": val_rmse
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        param = request.get_json()
        user_id = param.get('user_id')
        item_id = param.get('item_id')
        
        data = pd.read_csv('./data/dummy.csv', parse_dates=['tanggal'])  # Ambil data dari server Firebase
        grouped_data = { (user, item): df.assign(tanggal=pd.to_datetime(df['tanggal'])) for (user, item), df in data.groupby(['user_id', 'item_id']) }
                
        prediction = predict_sales(user_id, item_id, grouped_data[(user_id, item_id)], future_days=7)
        return jsonify({'forecast': prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)