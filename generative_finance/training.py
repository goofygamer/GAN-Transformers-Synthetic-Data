import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class TimeGANTrainer:
    """A class to encapsulate the WGAN-GP training logic."""

    def __init__(self, models, latent_dim, learning_rates, gp_weight=10.0):
        self.encoder = models['encoder']
        self.recovery = models['recovery']
        self.generator = models['generator']
        self.critic = models['discriminator'] # Renamed for clarity
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight

        # WGAN typically uses Adam with specific beta_1
        self.encoder_optimizer = Adam(learning_rate=learning_rates['autoencoder'], beta_1=0.5, beta_2=0.9)
        self.recovery_optimizer = Adam(learning_rate=learning_rates['autoencoder'], beta_1=0.5, beta_2=0.9)
        self.generator_optimizer = Adam(learning_rate=learning_rates['gan']['generator'], beta_1=0.5, beta_2=0.9)
        self.critic_optimizer = Adam(learning_rate=learning_rates['gan']['discriminator'], beta_1=0.5, beta_2=0.9)
        self.supervisor_optimizer = Adam(learning_rate=learning_rates['supervisor'], beta_1=0.5, beta_2=0.9)

    # --- WGAN Loss Functions ---
    def _critic_loss(self, real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def _generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def _gradient_penalty(self, real_data, fake_data):
        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_data - real_data
        interpolated = real_data + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    # --- Training Steps ---
    @tf.function
    def _train_autoencoder_step(self, X):
        with tf.GradientTape() as tape:
            H = self.encoder(X)
            X_tilde = self.recovery(H)
            e_loss = tf.sqrt(tf.reduce_mean(tf.square(X - X_tilde))) * 10
        var_list = self.encoder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        self.encoder_optimizer.apply_gradients(zip(gradients, var_list))
        return e_loss

    @tf.function
    def _train_supervisor_step(self, X):
        with tf.GradientTape() as tape:
            H = self.encoder(X)
            H_hat_supervised = self.generator(H)
            g_loss_s = tf.reduce_mean(tf.square(H - H_hat_supervised))
        var_list = self.generator.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        self.supervisor_optimizer.apply_gradients(zip(gradients, var_list))
        return g_loss_s
        
    @tf.function
    def _train_critic_step(self, X, Z):
        with tf.GradientTape() as tape:
            H_real = self.encoder(X, training=True)
            H_fake = self.generator(Z, training=True)

            real_output = self.critic(H_real, training=True)
            fake_output = self.critic(H_fake, training=True)

            critic_loss = self._critic_loss(real_output, fake_output)
            gp = self._gradient_penalty(H_real, H_fake)
            total_critic_loss = critic_loss + gp * self.gp_weight
        
        gradients = tape.gradient(total_critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        return total_critic_loss

    @tf.function
    def _train_generator_step(self, Z):
        with tf.GradientTape() as tape:
            H_fake = self.generator(Z, training=True)
            fake_output = self.critic(H_fake, training=True)
            g_loss = self._generator_loss(fake_output)
        
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return g_loss
        
    # --- Main Training Loop ---
    def train(self, data, epochs, batch_size, critic_steps=5):
        seq_len = data.shape[1]
        for epoch in range(epochs):
            for X_batch in tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=len(data)).batch(batch_size):
                # Train Critic more than Generator
                for _ in range(critic_steps):
                    Z_batch = tf.random.normal(shape=(tf.shape(X_batch)[0], seq_len, self.latent_dim))
                    step_c_loss = self._train_critic_step(X_batch, Z_batch)
                
                # Train Generator
                Z_batch = tf.random.normal(shape=(tf.shape(X_batch)[0], seq_len, self.latent_dim))
                step_g_loss = self._train_generator_step(Z_batch)

                # Train Autoencoder and Supervisor (less frequently if desired)
                step_e_loss = self._train_autoencoder_step(X_batch)
                step_g_loss_s = self._train_supervisor_step(X_batch)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, D_loss: {step_c_loss:.4f}, G_loss: {step_g_loss:.4f}, "
                      f"E_loss: {step_e_loss:.4f}, G_loss_S: {step_g_loss_s:.4f}")