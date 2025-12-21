import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Embedding, Input, Reshape
from tensorflow.keras.models import Model

# --- 1. Define Parameters ---
SEQ_LENGTH = 4  # Reduced to work with limited data
FEATURE_DIM = 3  # Price, Volatility, Volume
LATENT_DIM = 100 # Size of the random noise vector
EPOCHS = 500 # Reduced for quicker training
BATCH_SIZE = 64
NUM_CLASSES = 3  # e.g., Normal, Bullish, Bearish (for black swan)

# --- 1.5. Load and Preprocess Data ---
def load_and_preprocess_data(data_path='data/adam_market_baseline.json'):
    """Loads, parses, and preprocesses the financial data."""
    with open(data_path, 'r') as f:
        lines = f.readlines()
        # Filter out comment lines
        json_lines = [line for line in lines if not line.strip().startswith('#')]
        json_content = "".join(json_lines).strip()

        # The provided JSON has two identical blocks, so we read one
        # The file contains two JSON objects. We want the second, more complete one.
        decoder = json.JSONDecoder()
        pos = 0
        json_objects = []
        while pos < len(json_content):
            try:
                obj, end_pos = decoder.raw_decode(json_content[pos:])
                json_objects.append(obj)
                pos += end_pos
                # Skip whitespace and newlines
                while pos < len(json_content) and json_content[pos].isspace():
                    pos += 1
            except json.JSONDecodeError:
                # Assuming this is the end of the content
                break
        
        if len(json_objects) > 1:
            data = json_objects[1] # We want the second object
        elif len(json_objects) == 1:
            data = json_objects[0]
        else:
            raise ValueError("No JSON objects found in the file.")

    sp500_data = data['market_baseline']['data_modules']['asset_classes']['equities']['stock_indices']['sp500']

    prices = pd.DataFrame(sp500_data['historical_prices']).set_index('date')['price']
    volatility = pd.DataFrame(sp500_data['volatility']).set_index('date')['volatility']
    volume = pd.DataFrame(sp500_data['trading_volume']).set_index('date')['volume']

    df = pd.concat([prices, volatility, volume], axis=1).rename(columns={'price': 'Price', 'volatility': 'Volatility', 'volume': 'Volume'})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create sequences
    sequences = []
    for i in range(len(scaled_data) - SEQ_LENGTH + 1):
        sequences.append(scaled_data[i:i + SEQ_LENGTH])

    return np.array(sequences), scaler


# --- 2. Build the Generator ---
def build_generator():
    noise_input = Input(shape=(LATENT_DIM,))
    label_input = Input(shape=(1,))

    label_embedding = Embedding(NUM_CLASSES, 50)(label_input)
    label_embedding = Dense(LATENT_DIM)(label_embedding)
    label_embedding = Reshape((LATENT_DIM,))(label_embedding)

    merged_input = Concatenate()([noise_input, label_embedding])

    x = Dense(128, activation='relu')(merged_input)
    x = Dense(SEQ_LENGTH * FEATURE_DIM, activation='relu')(x)
    x = Reshape((SEQ_LENGTH, FEATURE_DIM))(x)

    x = LSTM(64, return_sequences=True)(x)
    output = Dense(FEATURE_DIM, activation='sigmoid')(x)

    model = Model([noise_input, label_input], output)
    model.summary()
    return model

# --- 3. Build the Discriminator ---
def build_discriminator():
    data_input = Input(shape=(SEQ_LENGTH, FEATURE_DIM))
    label_input = Input(shape=(1,))

    label_embedding = Embedding(NUM_CLASSES, 50)(label_input)
    label_embedding = Dense(SEQ_LENGTH * FEATURE_DIM)(label_embedding)
    label_embedding = Reshape((SEQ_LENGTH, FEATURE_DIM))(label_embedding)

    merged_input = Concatenate()([data_input, label_embedding])

    x = LSTM(64)(merged_input)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model([data_input, label_input], output)
    model.summary()
    return model

# --- 4. Build and Compile the Combined GAN Model ---
def build_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False

    noise_input = Input(shape=(LATENT_DIM,))
    label_input = Input(shape=(1,))

    generated_data = generator([noise_input, label_input])
    gan_output = discriminator([generated_data, label_input])

    gan = Model([noise_input, label_input], gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    gan.summary()
    return gan

# --- 5. Training Loop ---
def train_gan(gan, generator, discriminator, real_data):
    # Create dummy labels for the real data.
    # NOTE: For a real-world application, this is where you would use actual
    # labeled data that corresponds to different market conditions (e.g., normal,
    # bullish, bearish/black swan). The random labels are a placeholder for this PoC.
    real_labels = np.random.randint(0, NUM_CLASSES, real_data.shape[0])

    for epoch in range(EPOCHS):
        # --- Train Discriminator ---
        # Select a random batch of real data and labels
        idx = np.random.randint(0, real_data.shape[0], BATCH_SIZE)
        real_batch = real_data[idx]
        real_batch_labels = real_labels[idx]

        # Generate a batch of fake data and labels
        noise = np.random.randn(BATCH_SIZE, LATENT_DIM)
        fake_labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
        fake_batch = generator.predict([noise, fake_labels])

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch([real_batch, real_batch_labels], np.ones((BATCH_SIZE, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_batch, fake_labels], np.zeros((BATCH_SIZE, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # --- Train Generator ---
        noise = np.random.randn(BATCH_SIZE, LATENT_DIM)
        fake_labels_for_generator = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
        g_loss = gan.train_on_batch([noise, fake_labels_for_generator], np.ones((BATCH_SIZE, 1)))

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# --- 6. Main Execution ---
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    real_data, scaler = load_and_preprocess_data()

    if real_data.size == 0:
        print("Not enough data to create sequences. Exiting.")
    else:
        print("Building Generator...")
        generator = build_generator()

        print("\nBuilding Discriminator...")
        discriminator = build_discriminator()

        print("\nBuilding Combined GAN...")
        gan = build_gan(generator, discriminator)

        print("\n--- Starting GAN Training ---")
        train_gan(gan, generator, discriminator, real_data)
        print("--- Finished GAN Training ---")

        # --- Generate and Save Synthetic Data ---
        print("\nGenerating synthetic 'black swan' scenario...")
        noise = np.random.randn(1, LATENT_DIM)
        # Assuming label '2' corresponds to a bearish/black swan event
        black_swan_label = np.array([2]) 
        generated_sequence = generator.predict([noise, black_swan_label])

        # Inverse transform the generated data
        inversed_data = scaler.inverse_transform(generated_sequence.reshape(-1, FEATURE_DIM))

        # Reshape and create a DataFrame
        df = pd.DataFrame(inversed_data, columns=['Price', 'Volatility', 'Volume'])

        # Ensure the 'downloads' directory exists
        if not os.path.exists('downloads'):
            os.makedirs('downloads')

        output_path = 'downloads/synthetic_black_swan_scenario.csv'
        df.to_csv(output_path, index=False)

        print(f"Successfully generated and saved synthetic data to {output_path}")
        print("\nPoC Script Execution Complete.")
