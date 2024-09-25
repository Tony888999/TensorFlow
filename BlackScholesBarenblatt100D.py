import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from FBSNN import FBSNN
class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers):
        super().__init__(Xi, T, M, N, D, layers)
        """
        Initialize the Black-Scholes-Barenblatt model.

        Parameters:
        - Xi: Initial condition for X (initial stock prices).
        - T: Terminal time.
        - M: Number of trajectories (batch size).
        - N: Number of time steps.
        - D: Dimension (number of assets).
        - layers: Neural network architecture.

        The model solves the Black-Scholes-Barenblatt PDE using the deep BSDE method.
        """

    def phi_tf(self, t, X, Y, Z):
        """
        Represents the nonlinearity \( \varphi(t, X_t, Y_t, Z_t) \) in the BSDE.

        For the Black-Scholes-Barenblatt equation:
        \[
        \varphi(t, X_t, Y_t, Z_t) = 0
        \]
        """
        return 0.05 * (Y - tf.reduce_sum(X * Z, axis=1, keepdims=True))

    def g_tf(self, X):
        """
        Represents the terminal payoff function \( g(X_T) \).

        For a European put option:
        \[
        g(X_T) = \max(K - \frac{1}{D} \sum_{i=1}^D X_T^{(i)}, 0)
        \]
        """
        return tf.reduce_sum(X ** 2, axis=1, keepdims=True)

    def mu_tf(self, t, X, Y, Z):
        """
        Represents the drift coefficient \( \mu(t, X_t, Y_t, Z_t) \) in the forward SDE.

        For this problem, \( \mu(t, X_t, Y_t, Z_t) = 0 \).
        """
        return tf.zeros_like(X)

    def sigma_tf(self, t, X, Y):
        # Since tf.linalg.diag requires 1D tensors, we need to vectorize over the batch
        # We use tf.map_fn to apply the function over the batch dimension
        def diag_fn(x):
            return tf.linalg.diag(0.4 * x)
        sigma = tf.map_fn(diag_fn, X)
        return sigma  # M x D x D


# Define a learning rate schedule function
def learning_rate_schedule(it):
    if it < 20000:
        return 1e-3
    elif it < 50000:
        return 1e-4
    elif it < 80000:
        return 1e-5
    else:
        return 1e-6





# Main execution block
if __name__ == "__main__":
    M = 100  # number of trajectories (batch size)
    N = 50   # number of time snapshots
    D = 100  # number of dimensions

    layers = [D + 1] + 4 * [256] + [1]

    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
    T = 1.0

    # Training
    model = BlackScholesBarenblatt(Xi, T, M, N, D, layers)

    # Train the model using the learning rate schedule
    model.train(N_Iter=20, learning_rate_schedule=learning_rate_schedule)

    # PLOT RESULTS
    t_test, W_test = model.fetch_minibatch()

    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    def u_exact(t, X):  # (N+1) x 1, (N+1) x D
        r = 0.05
        sigma_max = 0.4
        return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, axis=1, keepdims=True)  # (N+1) x 1

    M_test = M  # Number of test samples
    Y_test = u_exact(
        np.reshape(t_test[0:M_test, :, :], [-1, 1]),
        np.reshape(X_pred[0:M_test, :, :], [-1, D])
    )
    Y_test = np.reshape(Y_test, [M_test, -1, 1])

    samples = 5

    plt.figure()
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('100-dimensional Black-Scholes-Barenblatt')
    plt.legend()
    plt.savefig('./figures/BSB_Sep21_50')
    plt.show()

    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    mean_errors = np.mean(errors, axis=0)
    std_errors = np.std(errors, axis=0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title('100-dimensional Black-Scholes-Barenblatt')
    plt.legend()
    plt.savefig('./figures/BSB_Sep21_50_errors')
    plt.show()
