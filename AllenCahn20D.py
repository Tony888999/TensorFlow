import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from FBSNN import FBSNN


class AllenCahn(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers):
        super().__init__(Xi, T, M, N, D, layers)
        """
         Initialize the Allen-Cahn model.

         Parameters:
         - Xi: Initial condition for X (vector of zeros).
         - T: Terminal time.
         - M: Number of trajectories (batch size).
         - N: Number of time steps.
         - D: Dimension of the problem.
         - layers: Neural network architecture.

         The model solves the Allen-Cahn PDE using the deep BSDE method.
         """

    def phi_tf(self, t, X, Y, Z):
        """
        Represents the nonlinearity \( \varphi(t, X_t, Y_t, Z_t) \) in the BSDE.

        For the Allen-Cahn equation:
        \[
        \varphi(t, X_t, Y_t, Z_t) = -Y_t + Y_t^3
        \]
        """
        return -Y + Y ** 3

    def g_tf(self, X):
        """
        Represents the terminal condition \( g(X_T) \) in the BSDE.

        For the Allen-Cahn equation:
        \[
        g(X_T) = \frac{1}{2 + 0.4 \| X_T \|^2}
        \]
        """
        return 1.0 / (2.0 + 0.4 * tf.reduce_sum(X ** 2, axis=1, keepdims=True))

    def mu_tf(self, t, X, Y, Z):
        """
        Represents the drift coefficient \( \mu(t, X_t, Y_t, Z_t) \) in the forward SDE.

        For this problem, \( \mu(t, X_t, Y_t, Z_t) = 0 \).
        """
        return tf.zeros_like(X)

    def sigma_tf(self, t, X, Y):
        """
        Represents the diffusion coefficient \( \sigma(t, X_t, Y_t) \) in the forward SDE.

        For this problem, \( \sigma(t, X_t, Y_t) = I \) (identity matrix).
        """
        batch_size = tf.shape(X)[0]
        D = tf.shape(X)[1]
        sigma = tf.eye(D, batch_shape=[batch_size])
        return sigma

if __name__ == "__main__":
    M = 100  # number of trajectories (batch size)
    N = 15   # number of time snapshots
    D = 20   # number of dimensions

    layers = [D + 1] + 4 * [256] + [1]

    T = 0.3
    Xi = np.zeros([1, D], dtype=np.float32)

    # Define the learning rate schedule
    boundaries = [20000, 50000, 80000]
    values = [1e-3, 1e-4, 1e-5, 1e-6]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    # Create an optimizer with the learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    # Initialize the model
    model = AllenCahn(Xi, T, M, N, D, layers)

    # Assign the optimizer with the learning rate schedule to the model
    model.optimizer = optimizer

    # Total number of iterations
    N_Iter = 50  # Sum of previous iterations: 2e4 + 3e4 + 3e4 + 2e4

    # Training
    model.train(N_Iter)

    # Fetch test data
    t_test, W_test = model.fetch_minibatch()

    # Make predictions
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    samples = 5

    Y_test_terminal = 1.0 / (2.0 + 0.4 * np.sum(X_pred[:, -1, :] ** 2, axis=1, keepdims=True))

    plt.figure()
    for i in range(samples):
        plt.plot(t_test[i, :, 0], Y_pred[i, :, 0], 'b')
    plt.plot(t_test[0, :, 0], Y_pred[0, :, 0], 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[:samples, -1, 0], Y_test_terminal[:samples, 0], 'ks', label='$Y_T = u(T,X_T)$')
    plt.plot([0], [0.30879], 'ko', label='$Y_0 = u(0,X_0)$')
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('20-dimensional Allen-Cahn')
    plt.legend()
    plt.savefig('./figures/AC_Seq21_15')
    plt.show()
