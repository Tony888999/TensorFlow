import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

# https://maziarraissi.github.io/FBSNNs/



class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, layers):
        self.Xi = Xi.astype(np.float32)  # Initial condition for X
        self.T = T                       # Terminal time
        self.M = M                       # Batch size
        self.N = N                       # Number of time steps
        self.D = D                       # Dimension of X
        self.layers = layers             # Neural network architecture

        # Initialize neural network parameters (weights and biases)
        self.weights = []
        self.biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            in_dim = layers[l]
            out_dim = layers[l + 1]
            # Xavier initialization for weights
            std_dev = np.sqrt(2 / (in_dim + out_dim))
            W = tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=std_dev), dtype=tf.float32)
            b = tf.Variable(tf.zeros([1, out_dim], dtype=tf.float32))
            self.weights.append(W)
            self.biases.append(b)

        # Optimizer (Adam) with default learning rate
        self.optimizer = tf.keras.optimizers.Adam()

    def neural_net(self, X):
        """
         Defines the neural network structure.

         Input:
         - X: Concatenated input of t and X_t.

         Output:
         - Y: Approximation of u(t, X_t).

         The neural network approximates the solution \( u(t, x) \) of the PDE.
         """
        H = X  # Input layer
        # Hidden layers with sine activation functions
        for l in range(len(self.weights) - 1):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.sin(tf.matmul(H, W) + b)
        # Output layer
        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.matmul(H, W) + b
        return Y

    def net_u(self, t, X):
        """
        Computes the network output u(t, X_t) and its gradient w.r.t X.

        This corresponds to approximating \( Y_t = u(t, X_t) \) and \( Z_t = D_u(t, X_t) \)
        in the BSDE (Equation (3) in the paper).

        We use automatic differentiation to compute the gradient.
        """

        with tf.GradientTape() as tape:
            tape.watch(X)
            u = self.neural_net(tf.concat([t, X], axis=1))
        Du = tape.gradient(u, X)
        return u, Du # Y_t, Z_t

    def Dg_tf(self, X):
        """
        Computes the gradient of the terminal condition g(X_T) w.r.t X_T.

        Used in enforcing the terminal condition for Z_T (Equation (3) in the paper).

        """
        with tf.GradientTape() as tape:
            tape.watch(X)
            g = self.g_tf(X)
        Dg = tape.gradient(g, X)
        return Dg

    def loss_function(self, t, W):
        # t: M x (N+1) x 1, W: M x (N+1) x D
        loss = 0
        X_list = []
        Y_list = []

        M = tf.shape(t)[0]
        N = self.N
        D = self.D

        t0 = t[:, 0, :]  # M x 1
        W0 = W[:, 0, :]  # M x D
        X0 = tf.tile(self.Xi, [M, 1])  # M x D
        Y0, Z0 = self.net_u(t0, X0)  # M x 1, M x D

        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(N):
            t1 = t[:, n + 1, :]  # M x 1
            W1 = W[:, n + 1, :]  # M x D
            dt = t1 - t0  # M x 1
            dW = W1 - W0  # M x D

            mu = self.mu_tf(t0, X0, Y0, Z0)  # M x D
            sigma = self.sigma_tf(t0, X0, Y0)  # M x D x D

            # Compute sigma * dW
            sigma_dW = tf.matmul(sigma, tf.expand_dims(dW, -1))  # M x D x 1
            sigma_dW = tf.squeeze(sigma_dW, axis=-1)  # M x D

            X1 = X0 + mu * dt + sigma_dW  # M x D

            phi = self.phi_tf(t0, X0, Y0, Z0)  # M x 1

            # Compute Z0 * sigma * dW
            sigma_Z_dW = tf.reduce_sum(Z0 * sigma_dW, axis=1, keepdims=True)  # M x 1

            Y1_tilde = Y0 + phi * dt + sigma_Z_dW  # M x 1

            Y1, Z1 = self.net_u(t1, X1)  # M x 1, M x D

            loss += tf.reduce_sum(tf.square(Y1 - Y1_tilde))  # Sum over batch

            # Update variables for next iteration
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

            X_list.append(X0)
            Y_list.append(Y0)

        # Terminal condition losses
        loss += tf.reduce_sum(tf.square(Y1 - self.g_tf(X1)))
        loss += tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1)))

        X = tf.stack(X_list, axis=1)  # M x (N+1) x D
        Y = tf.stack(Y_list, axis=1)  # M x (N+1) x 1

        Y0_pred = Y_list[0]  # M x 1

        return loss, X, Y, Y0_pred

    # using mean
    # def loss_function(self, t, W):
    #     loss = 0
    #     X_list = []
    #     Y_list = []
    #
    #     t0 = t[:, 0, :]
    #     X0 = tf.tile(self.Xi, [self.M, 1])
    #     Y0, Z0 = self.net_u(t0, X0)
    #
    #     X_list.append(X0)
    #     Y_list.append(Y0)
    #
    #     for n in range(self.N):
    #         dt = t[:, n+1, :] - t[:, n, :]
    #         dW = W[:, n+1, :] - W[:, n, :]
    #         t1 = t[:, n+1, :]
    #
    #         mu = self.mu_tf(t0, X0, Y0, Z0)
    #         sigma = self.sigma_tf(t0, X0, Y0)
    #
    #         sigma_dW = tf.matmul(sigma, tf.expand_dims(dW, -1))[:, :, 0]
    #         X1 = X0 + mu * dt + sigma_dW
    #         phi = self.phi_tf(t0, X0, Y0, Z0)
    #         Y1_tilde = Y0 + phi * dt + tf.reduce_sum(Z0 * sigma_dW, axis=1, keepdims=True)
    #
    #         Y1, Z1 = self.net_u(t1, X1)
    #
    #         loss += tf.reduce_mean(tf.square(Y1 - Y1_tilde))
    #
    #         t0 = t1
    #         X0 = X1
    #         Y0 = Y1
    #         Z0 = Z1
    #
    #         X_list.append(X0)
    #         Y_list.append(Y0)
    #
    #     g_X1 = self.g_tf(X1)
    #     Dg_X1 = self.Dg_tf(X1)
    #     loss += tf.reduce_mean(tf.square(Y1 - g_X1))
    #     loss += tf.reduce_mean(tf.square(Z1 - Dg_X1))
    #
    #     X = tf.stack(X_list, axis=1)
    #     Y = tf.stack(Y_list, axis=1)
    #     Y0_pred = Y_list[0]
    #
    #     return loss, X, Y, Y0_pred

    def fetch_minibatch(self):
        """
        Generates a minibatch of time steps and Brownian motion increments.

        This corresponds to simulating sample paths for the forward SDE.

        The time steps are:
        \[
        t^n = n \Delta t, \quad \Delta t = \frac{T}{N}
        \]
        The Brownian increments are:
        \[
        \Delta W^n \sim \mathcal{N}(0, \Delta t)
        \]
        """

        dt = self.T / self.N
        t = np.linspace(0, self.T, self.N + 1)
        t = np.tile(t.reshape(1, -1, 1), (self.M, 1, 1)).astype(np.float32)

        dW = np.sqrt(dt) * np.random.randn(self.M, self.N + 1, self.D).astype(np.float32)
        dW[:, 0, :] = 0 # W_0 = 0
        W = np.cumsum(dW, axis=1) # Cumulative sum to get W_t
        return t, W

    def train(self, N_Iter, learning_rate_schedule=None):
        """
        Trains the neural network using the loss function.
        Parameters:
        - N_Iter: Number of training iterations.
        """

        for it in range(N_Iter):
            if learning_rate_schedule is not None:
                # Update the learning rate according to the schedule
                lr = learning_rate_schedule(it)
                self.optimizer.learning_rate.assign(lr)

            t_batch, W_batch = self.fetch_minibatch()

            with tf.GradientTape() as tape:
                loss_value, _, _, Y0_pred = self.loss_function(t_batch, W_batch)

            grads = tape.gradient(loss_value, self.weights + self.biases)
            self.optimizer.apply_gradients(zip(grads, self.weights + self.biases))

            if it % 10 == 0:
                print(
                    f'It: {it}, Loss: {loss_value.numpy():.3e}, Y0: {Y0_pred.numpy().mean():.3f}, Learning Rate: {self.optimizer.learning_rate.numpy():.1e}')

    def predict(self, Xi_star, t_star, W_star):
        """
        Makes predictions using the trained neural network.

        Returns the predicted paths of X and Y.
        """

        Xi_star = Xi_star.astype(np.float32)
        t_star = t_star.astype(np.float32)
        W_star = W_star.astype(np.float32)

        _, X_pred, Y_pred, _ = self.loss_function(t_star, W_star)
        return X_pred.numpy(), Y_pred.numpy()

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        """
        Abstract method for the nonlinearity \( \varphi(t, X_t, Y_t, Z_t) \) in the BSDE.

        Must be implemented in subclasses based on the specific PDE.
        """
        pass

    @abstractmethod
    def g_tf(self, X):
        """
        Abstract method for the terminal condition \( g(X_T) \).

        Must be implemented in subclasses based on the specific PDE.
        """
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        """
        Abstract method for the drift coefficient \( \mu(t, X_t, Y_t, Z_t) \) in the forward SDE.

        Must be implemented in subclasses based on the specific PDE.
        """
        pass

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        """
        Abstract method for the diffusion coefficient \( \sigma(t, X_t, Y_t) \) in the forward SDE.

        Must be implemented in subclasses based on the specific PDE.
        """
        pass
