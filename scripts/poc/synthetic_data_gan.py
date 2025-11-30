import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import os

# --- 1. Define Parameters ---
SEQ_LENGTH = 30  # Number of time steps in a sequence
FEATURE_DIM = 5  # O, H, L, C, V
LATENT_DIM = 100 # Size of the random noise vector
EPOCHS = 5000
BATCH_SIZE = 64

# --- 2. Build the Generator ---
def build_generator():
    noise_input = Input(shape=(LATENT_DIM,))

    x = Dense(128, activation='relu')(noise_input)
    x = Dense(SEQ_LENGTH * FEATURE_DIM, activation='relu')(x)
    x = Reshape((SEQ_LENGTH, FEATURE_DIM))(x)

    # Using LSTM to learn temporal patterns, even from noise
    x = LSTM(64, return_sequences=True)(x)
    output = Dense(FEATURE_DIM, activation='sigmoid')(x) # Sigmoid to keep output between 0 and 1

    model = Model(noise_input, output)
    model.summary()
    return model

# --- 3. Build the Discriminator ---
def build_discriminator():
    data_input = Input(shape=(SEQ_LENGTH, FEATURE_DIM))

    x = LSTM(64)(data_input)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x) # Sigmoid for binary classification (real/fake)

    model = Model(data_input, output)
    model.summary()
    return model

# --- 4. Build and Compile the Combined GAN Model ---
def build_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False # Freeze discriminator during combined training

    gan_input = Input(shape=(LATENT_DIM,))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)

    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    gan.summary()
    return gan

# --- 5. Training Loop ---
def train_gan(gan, generator, discriminator):
    # For this PoC, we use random data as our "real" data to prove the concept
    # In a real scenario, this would be loaded and preprocessed historical data
    real_data = np.random.rand(1000, SEQ_LENGTH, FEATURE_DIM)

    for epoch in range(EPOCHS):
        # --- Train Discriminator ---
        # Select a random batch of real data
        idx = np.random.randint(0, real_data.shape[0], BATCH_SIZE)
        real_batch = real_data[idx]

        # Generate a batch of fake data
        noise = np.random.randn(BATCH_SIZE, LATENT_DIM)
        fake_batch = generator.predict(noise)

        # Train the discriminator on real and fake data
        d_loss_real = discriminator.train_on_batch(real_batch, np.ones((BATCH_SIZE, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((BATCH_SIZE, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # --- Train Generator ---
        noise = np.random.randn(BATCH_SIZE, LATENT_DIM)
        # The generator is trained to make the discriminator label its output as real
        g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# --- 6. Main Execution ---
if __name__ == "__main__":
    print("Building Generator...")
    generator = build_generator()

    print("\nBuilding Discriminator...")
    discriminator = build_discriminator()

    print("\nBuilding Combined GAN...")
    gan = build_gan(generator, discriminator)

    print("\n--- Starting GAN Training ---")
    train_gan(gan, generator, discriminator)
    print("--- Finished GAN Training ---")

    # --- Generate and Save Synthetic Data ---
    print("\nGenerating synthetic data...")
    noise = np.random.randn(1, LATENT_DIM)
    generated_sequence = generator.predict(noise)

    # Reshape and create a DataFrame
    generated_data_flat = generated_sequence.reshape(-1, FEATURE_DIM)
    df = pd.DataFrame(generated_data_flat, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Ensure the 'downloads' directory exists
    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    output_path = 'downloads/synthetic_stock_data.csv'
    df.to_csv(output_path, index=False)

    print(f"Successfully generated and saved synthetic data to {output_path}")
    print("\nPoC Script Execution Complete.")
