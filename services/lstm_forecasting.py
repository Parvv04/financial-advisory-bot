from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from .config import LOOKBACK
import matplotlib.pyplot as plt




logger = logging.getLogger(__name__)

def create_dataset(dataset, target_cols, step):
    X, y = [], []
    for i in range(step, len(dataset)):
        X.append(dataset.iloc[i-step:i].values)
        y.append([dataset.iloc[i, dataset.columns.get_loc(col)] for col in target_cols])
    return np.array(X), np.array(y)

def prepare_model(symbols, stock_data, macro, lookback=30):
    print("Starting enhanced model preparation...")
    if stock_data is None or stock_data.empty or macro is None or macro.empty:
        print("[Error] Missing stock or macro data.")
        return None

    combined = pd.concat([stock_data, macro], axis=1).dropna()

    target_cols = symbols
    feature_cols = [col for col in combined.columns if col not in target_cols]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_split = int(len(combined) * 0.9)

    scaled_X = pd.DataFrame(
        scaler_X.fit_transform(combined[feature_cols]),
        columns=feature_cols,
        index=combined.index
    )
    scaled_y = pd.DataFrame(
        scaler_y.fit_transform(combined[target_cols]),
        columns=target_cols,
        index=combined.index
    )

    scaled_combined = pd.concat([scaled_X, scaled_y], axis=1)

    X, y = [], []
    for i in range(lookback, len(scaled_combined)):
        X.append(scaled_combined.iloc[i-lookback:i].values)
        y.append(scaled_y.iloc[i].values)

    X, y = np.array(X), np.array(y)

    split = int(0.9 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    from tensorflow.keras import Input
    model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(target_cols))
])



    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )

    print("[OK] Enhanced model trained.")
    return model, scaler_X, scaler_y, scaled_combined, X_test, target_cols, y_test, split

def predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("[OK] Predicting...")
    predictions = model.predict(X_test)

    if predictions.shape != y_test.shape:
        raise ValueError("Prediction shape mismatch!")

    # Inverse transform predictions and actuals to get original prices
    pred_unscaled = pd.DataFrame(
        scaler_y.inverse_transform(predictions),
        columns=target_cols
    )

    actual_unscaled = pd.DataFrame(
        scaler_y.inverse_transform(y_test),
        columns=target_cols
    )

    # Ensure proper index alignment
    test_data_index = combined_scaled.index[train_size + LOOKBACK:]
    pred_unscaled.index = test_data_index
    actual_unscaled.index = test_data_index

    results = {}
    evaluation = {}

    for symbol in target_cols:
        predicted = pred_unscaled[symbol]
        actual = actual_unscaled[symbol]

        results[symbol] = {
            "predicted": predicted,
            "actual": actual
        }

        evaluation[symbol] = {
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MAE": mean_absolute_error(actual, predicted),
            "R2": r2_score(actual, predicted)
        }

    print("[OK] Prediction complete.")
    return results, evaluation

def evaluate_predictions(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size):
    predictions = model.predict(X_test)
    pred_unscaled = pd.DataFrame(scaler_y.inverse_transform(predictions), columns=target_cols)
    actual_unscaled = pd.DataFrame(scaler_y.inverse_transform(y_test), columns=target_cols)

    test_data_index = combined_scaled.index[train_size + LOOKBACK:]
    pred_unscaled.index = test_data_index
    actual_unscaled.index = test_data_index

    results = {}
    evaluation = {}

    for symbol in target_cols:
        predicted = pred_unscaled[symbol]
        actual = actual_unscaled[symbol]
        results[symbol] = {
            "predicted": predicted,
            "actual": actual
        }
        evaluation[symbol] = {
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MAE": mean_absolute_error(actual, predicted),
            "R2": r2_score(actual, predicted)
        }

    print("[OK] Evaluation complete.")
    return results, evaluation
