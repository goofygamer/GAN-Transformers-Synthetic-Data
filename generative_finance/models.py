from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

def build_autoencoder(seq_len: int, latent_dim: int, feature_dim: int):
    """Builds the autoencoder components (encoder and recovery).

    Args:
        seq_len (int): The length of the input sequences.
        hidden_dim (int): The dimensionality of the hidden space.
        num_layers (int): The number of GRU layers.

    Returns:
        tuple: A tuple containing the encoder and recovery models.
    """
    
    # Encoder: Maps a real sequence to a latent sequence
    encoder = Sequential(name="Encoder")
    encoder.add(GRU(units=latent_dim, return_sequences=True, input_shape=(seq_len, feature_dim)))
    encoder.add(GRU(units=latent_dim, return_sequences=True)) # Must return a sequence

    # Recovery: Maps a latent sequence back to a real sequence
    recovery = Sequential(name="Recovery")
    recovery.add(GRU(units=latent_dim, return_sequences=True, input_shape=(seq_len, latent_dim)))
    recovery.add(GRU(units=feature_dim, return_sequences=True, activation='sigmoid'))

    return encoder, recovery

def build_generator(seq_len: int, latent_dim: int):
    """Builds the corrected generator component.
    Args:
        seq_len (int): The length of the input sequences.
        latent_dim (int): The dimensionality of the latent space.

    Returns:
        A sequential model of the generator.
    """
    
    generator = Sequential(name="Generator")
    generator.add(GRU(units=latent_dim, return_sequences=True, input_shape=(seq_len, latent_dim)))
    generator.add(GRU(units=latent_dim, return_sequences=True, activation='sigmoid'))

    return generator

def build_discriminator(seq_len: int, latent_dim: int):
    """Builds the corrected generator component.
    Args:
        seq_len (int): The length of the input sequences.
        latent_dim (int): The dimensionality of the latent space.

    Returns:
        A sequential model of the discriminator.
    """
    
    discriminator = Sequential(name="Discriminator")
    discriminator.add(GRU(units=latent_dim, return_sequences=True, input_shape=(seq_len, latent_dim)))
    discriminator.add(GRU(units=latent_dim, return_sequences=False)) # Reduces sequence to a vector for classification
    discriminator.add(Dense(units=1, activation='sigmoid'))

    return discriminator