import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('preprocessed_nasa_logs.csv')

vectorizer = TfidfVectorizer(max_features=100)
df['description_tfidf'] = list(vectorizer.fit_transform(df['description']).toarray())

scaler = MinMaxScaler()
df[['year_issued', 'description_length']] = scaler.fit_transform(df[['year_issued', 'description_length']])

X = pd.concat([df[['year_issued', 'description_length']], pd.DataFrame(df['description_tfidf'].to_list())], axis=1)

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

def build_autoencoder(input_dim):
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')  # Output layer same size as input
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

input_dim = X_train.shape[1]
autoencoder = build_autoencoder(input_dim)

history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

X_test_pred = autoencoder.predict(X_test)
reconstruction_errors = np.mean(np.square(X_test - X_test_pred), axis=1)

threshold = np.percentile(reconstruction_errors, 95)
anomalies = X_test[reconstruction_errors > threshold]

print(f"Number of anomalies detected: {len(anomalies)}")

plt.hist(reconstruction_errors, bins=50)
plt.axvline(threshold, color='r', linestyle='--', label='Anomaly Threshold')
plt.title('Reconstruction Errors')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

fig = px.scatter(x=range(len(reconstruction_errors)), y=reconstruction_errors,
                 title='Reconstruction Errors for Test Set',
                 labels={'x':'Index', 'y':'Reconstruction Error'})
fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
fig.show()
