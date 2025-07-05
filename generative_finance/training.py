import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

class TimeGANTrainer:
    """An updated class to allow for flexible GAN training."""

    def __init__(self, models, latent_dim, learning_rates):
        self.encoder = models['encoder']
        self.recovery = models['recovery']
        self.generator = models['generator']
        self.discriminator = models['discriminator']
        self.latent_dim = latent_dim

        # Accept separate learning rates for generator and discriminator
        self.encoder_optimizer = Adam(learning_rate=learning_rates['autoencoder'])
        self.recovery_optimizer = Adam(learning_rate=learning_rates['autoencoder'])
        self.generator_optimizer = Adam(learning_rate=learning_rates['gan']['generator'])
        self.discriminator_optimizer = Adam(learning_rate=learning_rates['gan']['discriminator'])
        self.supervisor_optimizer = Adam(learning_rate=learning_rates['supervisor'])
        
        self.bce = BinaryCrossentropy()
        self.mse = MeanSquaredError()

    @tf.function
    def _train_autoencoder_step(self, X):
        with tf.GradientTape() as tape:
            H = self.encoder(X)
            X_tilde = self.recovery(H)
            e_loss = 10 * tf.sqrt(self.mse(X, X_tilde))

        var_list = self.encoder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        self.encoder_optimizer.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(self.mse(X, X_tilde))

    @tf.function
    def _train_supervisor_step(self, X):
        with tf.GradientTape() as tape:
            H = self.encoder(X)
            H_hat_supervised = self.generator(H)
            g_loss_s = self.mse(H, H_hat_supervised)

        var_list = self.generator.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        self.supervisor_optimizer.apply_gradients(zip(gradients, var_list))
        return g_loss_s

    # --- SPLIT GAN TRAINING FUNCTIONS ---

    @tf.function
    def _train_gan_discriminator_step(self, X, Z):
        """Trains only the discriminator for one step."""
        with tf.GradientTape() as tape:
            H = self.encoder(X, training=False) # Encoder is not trained here
            H_hat = self.generator(Z, training=False) # Generator is not trained here

            Y_real = self.discriminator(H)
            Y_fake = self.discriminator(H_hat)

            d_loss_real = self.bce(y_true=tf.ones_like(Y_real), y_pred=Y_real)
            d_loss_fake = self.bce(y_true=tf.zeros_like(Y_fake), y_pred=Y_fake)
            d_loss = d_loss_real + d_loss_fake

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(d_loss, var_list)
        self.discriminator_optimizer.apply_gradients(zip(gradients, var_list))
        return d_loss

    @tf.function
    def _train_gan_generator_step(self, X, Z):
        """Trains only the generator for one step."""
        with tf.GradientTape() as tape:
            H = self.encoder(X, training=False) # Encoder is not trained here
            H_hat = self.generator(Z)
            Y_fake = self.discriminator(H_hat, training=False) # Discriminator is not trained here
            
            g_loss_u = self.bce(y_true=tf.ones_like(Y_fake), y_pred=Y_fake)
            g_loss_v = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(H_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(H, [0])[1] + 1e-6)))
            g_loss = g_loss_u + 100 * g_loss_v

        var_list = self.generator.trainable_variables
        gradients = tape.gradient(g_loss, var_list)
        self.generator_optimizer.apply_gradients(zip(gradients, var_list))
        return g_loss_u, g_loss_v

    # --- UPDATED MAIN TRAINING LOOP ---

    def train(self, data, epochs, batch_size, g_train_steps=1):
        """The main training loop with adjustable generator training steps."""
        seq_len = data.shape[1]
        for epoch in range(epochs):
            for X_batch in tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=len(data)).batch(batch_size):
                Z_batch = tf.random.normal(shape=(tf.shape(X_batch)[0], seq_len, self.latent_dim))
                
                # 1. Train Autoencoder
                step_e_loss = self._train_autoencoder_step(X_batch)
                
                # 2. Train Supervisor
                step_g_loss_s = self._train_supervisor_step(X_batch)
                
                # 3. Train Discriminator (once per batch)
                step_d_loss = self._train_gan_discriminator_step(X_batch, Z_batch)

                # 4. Train Generator (multiple steps per batch)
                for _ in range(g_train_steps):
                    step_g_loss_u, step_g_loss_v = self._train_gan_generator_step(X_batch, Z_batch)
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, E_loss: {step_e_loss:.4f}, G_loss_S: {step_g_loss_s:.4f}, "
                      f"G_loss_U: {step_g_loss_u:.4f}, D_loss: {step_d_loss:.4f}")