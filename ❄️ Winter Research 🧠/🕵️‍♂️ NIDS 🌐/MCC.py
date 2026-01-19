import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Manually entered dataset
data = {
    "Filler type/content (wt%)": [
        "Epoxy", "Epoxy/CB", "Epoxy/CB", "Epoxy/CB",
        "Epoxy/SWCNT", "Epoxy/SWCNT", "Epoxy/SWCNT",
        "Epoxy/DWCNT", "Epoxy/DWCNT", "Epoxy/DWCNT",
        "Epoxy/DWCNT–NH2", "Epoxy/DWCNT–NH2", "Epoxy/DWCNT–NH2",
        "Epoxy/MWCNT", "Epoxy/MWCNT", "Epoxy/MWCNT",
        "Epoxy/MWCNT–NH2", "Epoxy/MWCNT–NH2", "Epoxy/MWCNT–NH2"
    ],
    "Content (wt%)": [
        0.0, 0.1, 0.3, 0.5, 0.05, 0.1, 0.3, 0.1, 0.3, 0.5,
        0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5
    ],
    "Young's modulus (MPa)": [
        2599, 2752, 2796, 2830, 2681, 2691, 2812, 2785, 2885, 2790,
        2610, 2944, 2978, 2780, 2765, 2609, 2884, 2819, 2820
    ],
    "Young's modulus ±": [
        81, 144, 34, 60, 80, 31, 90, 23, 88, 29, 104, 50, 24,
        40, 53, 13, 32, 45, 15
    ],
    "Ultimate tensile strength (MPa)": [
        63.80, 63.28, 63.13, 65.34, 65.84, 66.34, 67.28, 62.43, 67.77,
        67.66, 63.62, 67.02, 69.13, 62.97, 63.17, 61.52, 64.67, 63.64, 64.27
    ],
    "Ultimate tensile strength ±": [
        1.09, 0.85, 0.59, 0.82, 0.64, 1.11, 0.63, 1.08, 0.40,
        0.50, 0.68, 0.19, 0.61, 0.25, 0.13, 0.19, 0.13, 0.21, 0.42
    ],
    "Fracture toughness KIc (MPa m^1/2)": [
        0.65, 0.76, 0.86, 0.85, 0.72, 0.80, 0.73, 0.76, 0.85, 0.85,
        0.77, 0.92, 0.93, 0.79, 0.80, None, 0.81, 0.85, 0.84
    ],
    "Fracture toughness ±": [
        0.062, 0.030, 0.063, 0.034, 0.014, 0.041, 0.028, 0.043,
        0.031, 0.064, 0.024, 0.017, 0.030, 0.048, 0.028, None, 0.029, 0.018, 0.023
    ]
}

df = pd.DataFrame(data)

# Calculate bounds
df["Young's modulus Lower Bound (MPa)"] = df["Young's modulus (MPa)"] - df["Young's modulus ±"]
df["Young's modulus Upper Bound (MPa)"] = df["Young's modulus (MPa)"] + df["Young's modulus ±"]

df["Ultimate tensile strength Lower Bound (MPa)"] = df["Ultimate tensile strength (MPa)"] - df["Ultimate tensile strength ±"]
df["Ultimate tensile strength Upper Bound (MPa)"] = df["Ultimate tensile strength (MPa)"] + df["Ultimate tensile strength ±"]

df["Fracture toughness KIc Lower Bound (MPa m^1/2)"] = df["Fracture toughness KIc (MPa m^1/2)"] - df["Fracture toughness ±"]
df["Fracture toughness KIc Upper Bound (MPa m^1/2)"] = df["Fracture toughness KIc (MPa m^1/2)"] + df["Fracture toughness ±"]

# Encoding fillers
mechanical_properties = {
    (1.0, 0, 0, 0, 0, 0, 0, 2599, 81, 63.80, 1.09, 0.65, 0.062),
    (0, 0.1, 0, 0, 0, 0, 0, 2752, 144, 63.28, 0.85, 0.76, 0.030),
    (0, 0.3, 0, 0, 0, 0, 0, 2796, 34, 63.13, 0.59, 0.86, 0.063),
    (0, 0.5, 0, 0, 0, 0, 0, 2830, 60, 65.34, 0.82, 0.85, 0.034),
    (0, 0, 0.05, 0, 0, 0, 0, 2681, 80, 65.84, 0.64, 0.72, 0.014),
    (0, 0, 0.1, 0, 0, 0, 0, 2691, 31, 66.34, 1.11, 0.80, 0.041),
    (0, 0, 0.3, 0, 0, 0, 0, 2812, 90, 67.28, 0.63, 0.73, 0.028),
    (0, 0, 0, 0.1, 0, 0, 0, 2785, 23, 62.43, 1.08, 0.76, 0.043),
    (0, 0, 0, 0.3, 0, 0, 0, 2885, 88, 67.77, 0.40, 0.85, 0.031),
    (0, 0, 0, 0.5, 0, 0, 0, 2790, 29, 67.66, 0.50, 0.85, 0.064),
    (0, 0, 0, 0, 0.1, 0, 0, 2610, 104, 63.62, 0.68, 0.77, 0.024),
    (0, 0, 0, 0, 0.3, 0, 0, 2944, 50, 67.02, 0.19, 0.92, 0.017),
    (0, 0, 0, 0, 0.5, 0, 0, 2978, 24, 69.13, 0.61, 0.93, 0.030),
    (0, 0, 0, 0, 0, 0.1, 0, 2780, 40, 62.97, 0.25, 0.79, 0.048),
    (0, 0, 0, 0, 0, 0.3, 0, 2765, 53, 63.17, 0.21, 0.80, 0.028),
    (0, 0, 0, 0, 0, 0.5, 0, 2609, 13, 61.52, 0.19, np.nan, np.nan),
    (0, 0, 0, 0, 0, 0, 0.1, 2884, 32, 64.67, 0.13, 0.81, 0.029),
    (0, 0, 0, 0, 0, 0, 0.3, 2803, 45, 64.63, 0.21, 0.83, 0.028),
    (0, 0, 0, 0, 0, 0, 0.5, 2820, 15, 64.27, 0.32, 0.84, 0.028),
}

df1 = pd.DataFrame(data=mechanical_properties)
df2 = pd.concat([df1, df.reset_index(drop=True)], axis=1)

df2.rename(columns={
    0: "Epoxy", 1: "Epoxy_CB", 2: "Epoxy_SWCNT", 3: "Epoxy_DWCNT",
    4: "Epoxy_DWCNT_NH2", 5: "Epoxy_MWCNT", 6: "Epoxy_MWCNT_NH2"
}, inplace=True)

df_3 = df2[["Epoxy", "Epoxy_CB", "Epoxy_SWCNT", "Epoxy_DWCNT", "Epoxy_DWCNT_NH2", "Epoxy_MWCNT", "Epoxy_MWCNT_NH2",
            "Young's modulus Lower Bound (MPa)", "Young's modulus Upper Bound (MPa)",
            "Ultimate tensile strength Lower Bound (MPa)", "Ultimate tensile strength Upper Bound (MPa)",
            "Fracture toughness KIc Lower Bound (MPa m^1/2)", "Fracture toughness KIc Upper Bound (MPa m^1/2)"]]

# Bootstrapping
df_4 = pd.DataFrame(columns=df_3.columns)
bound = 3
bootstrap = 50000
for _ in range(bootstrap):
    temp = df_3.copy()
    for i in range(len(temp)):
        temp.loc[i, "Young's modulus (MPa)"] = np.random.normal(
            loc=(temp.iloc[i, 7] + temp.iloc[i, 8]) / 2,
            scale=(temp.iloc[i, 8] - temp.iloc[i, 7]) / bound
        )
        temp.loc[i, "Ultimate tensile strength (MPa)"] = np.random.normal(
            loc=(temp.iloc[i, 9] + temp.iloc[i, 10]) / 2,
            scale=(temp.iloc[i, 10] - temp.iloc[i, 9]) / bound
        )
        if not pd.isna(temp.iloc[i, 11]):
            temp.loc[i, "Fracture toughness KIc"] = np.random.normal(
                loc=(temp.iloc[i, 11] + temp.iloc[i, 12]) / 2,
                scale=(temp.iloc[i, 12] - temp.iloc[i, 11]) / bound
            )
    df_4 = pd.concat([df_4, temp], ignore_index=True)

# Select and clean final dataset
df_5 = df_4[['Epoxy', 'Epoxy_CB', 'Epoxy_SWCNT', 'Epoxy_DWCNT', 'Epoxy_DWCNT_NH2',
             'Epoxy_MWCNT', 'Epoxy_MWCNT_NH2', "Young's modulus (MPa)",
             'Ultimate tensile strength (MPa)', 'Fracture toughness KIc']]
df_5.dropna(inplace=True)

# Prepare inputs
X = df_5[['Epoxy_CB']]
y = df_5[["Young's modulus (MPa)", "Ultimate tensile strength (MPa)", "Fracture toughness KIc"]]

# Polynomial features
poly = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Standardization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# R² metric
def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# Build model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(y_train.shape[1])
])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9, staircase=True
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='mse', metrics=[r2_metric])

# Train model
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train_scaled, validation_split=0.2,
                    epochs=400, batch_size=16, callbacks=[early_stopping], verbose=1)

# Evaluation
loss, r2_score_avg = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f"\nOverall Test R2 Score: {r2_score_avg:.3f}")

y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

output_names = list(y.columns)

print("\nIndividual R2 Scores:")
for i, name in enumerate(output_names):
    r2 = r2_score(y_test_actual[:, i], y_pred[:, i])
    print(f"{name}: {r2:.3f}")

# Plot predictions
plt.figure(figsize=(18, 5))
for i, name in enumerate(output_names):
    plt.subplot(1, 3, i+1)
    plt.scatter(y_test_actual[:, i], y_pred[:, i], alpha=0.6, color='blue')
    plt.plot([min(y_test_actual[:, i]), max(y_test_actual[:, i])],
             [min(y_test_actual[:, i]), max(y_test_actual[:, i])], 'r--')
    plt.xlabel(f'Actual {name}')
    plt.ylabel(f'Predicted {name}')
    plt.title(f'{name}\nR2 Score: {r2_score(y_test_actual[:, i], y_pred[:, i]):.3f}')
    plt.grid(True)
plt.tight_layout()
plt.savefig("1")

# Training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['r2_metric'], label='Train R2')
plt.plot(history.history['val_r2_metric'], label='Val R2')
plt.title('R2 over Epochs')
plt.xlabel('Epoch')
plt.ylabel('R2 Score')
plt.legend()
plt.tight_layout()
plt.savefig("2")

# Predict on new values
def predict_multiple_cb_contents(cb_contents):
    cb_array = np.array(cb_contents).reshape(-1, 1)
    cb_poly = poly.transform(cb_array)
    cb_scaled = scaler_X.transform(cb_poly)
    pred_scaled = model.predict(cb_scaled)
    preds = scaler_y.inverse_transform(pred_scaled)

    return pd.DataFrame({
        'Epoxy_CB': cb_contents,
        "Young's modulus (MPa)": preds[:, 0],
        "Ultimate tensile strength (MPa)": preds[:, 1],
        "Fracture toughness KIc": preds[:, 2]
    })

cb_contents = [0.0, 0.1, 0.3, 0.5]
predictions_df = predict_multiple_cb_contents(cb_contents)
print("\nPredictions for CB content:")
print(predictions_df)

# Plot predictions
plt.figure(figsize=(18, 5))
for i, name in enumerate(output_names):
    plt.subplot(1, 3, i+1)
    plt.plot(predictions_df['Epoxy_CB'], predictions_df[name], 'bo-', label='Predicted')
    plt.scatter(X['Epoxy_CB'], y.iloc[:, i], color='red', alpha=0.6, label='Actual')
    plt.xlabel('Epoxy_CB Content')
    plt.ylabel(name)
    plt.title(f'{name} vs Epoxy_CB')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("3")
