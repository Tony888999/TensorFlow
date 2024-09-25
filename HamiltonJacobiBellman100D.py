import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from FBSNN import FBSNN


class HamiltonJacobiBellman(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers):
        super().__init__(Xi, T, M, N, D, layers)
        """
        Initialize the Hamilton-Jacobi-Bellman model.

        Parameters:
        - Xi: Initial condition for X.
        - T: Terminal time.
        - M: Number of trajectories (batch size).
        - N: Number of time steps.
        - D: Dimension.
        - layers: Neural network architecture.

        The model solves the HJB equation using the deep BSDE method.
        """

    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        """
         Represents the nonlinearity \( \varphi(t, X_t, Y_t, Z_t) \) in the BSDE.

         For the HJB equation:
         \[
         \varphi(t, X_t, Y_t, Z_t) = \| Z_t \|^2
         \]
         """
        return tf.reduce_sum(Z ** 2, axis=1, keepdims=True)  # M x 1

    def g_tf(self, X):  # M x D
        """
        Represents the terminal condition \( g(X_T) \) in the BSDE.

        For this problem:
        \[
        g(X_T) = \ln\left( 0.5 + 0.5 \| X_T \|^2 \right)
        \]
        """
        return tf.math.log(0.5 + 0.5 * tf.reduce_sum(X ** 2, axis=1, keepdims=True))  # M x 1

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        """
         Drift coefficient \( \mu(t, X_t, Y_t, Z_t) \) in the forward SDE.

         For this problem, \( \mu(t, X_t, Y_t, Z_t) = 0 \).
         """
        return tf.zeros_like(X)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        """
        Diffusion coefficient \( \sigma(t, X_t, Y_t) \) in the forward SDE.

        For this problem, \( \sigma(t, X_t, Y_t) = \sqrt{2} \cdot I \).
        """
        sqrt_2 = tf.sqrt(2.0)
        # Since the base class's sigma_tf returns identity matrices, we can define sigma as sqrt(2) * I
        batch_size = tf.shape(X)[0]
        D = tf.shape(X)[1]
        sigma = sqrt_2 * tf.eye(D, batch_shape=[batch_size])  # M x D x D
        return sigma


def learning_rate_schedule(it):
    if it < 20000:
        return 1e-3
    elif it < 50000:
        return 1e-4
    elif it < 80000:
        return 1e-5
    else:
        return 1e-6



# Define the exact solution function
def g(X):  # MC x NC x D
    return np.log(0.5 + 0.5 * np.sum(X ** 2, axis=2, keepdims=True))  # MC x NC x 1

def u_exact(t, X):  # t: NC, X: NC x D
    """
     Computes the exact solution u(t, X_t).
     This involves a Monte Carlo simulation.
     """
    MC = 10 ** 5
    NC = t.shape[0]
    D = X.shape[1]

    W = np.random.normal(size=(MC, NC, D))  # MC x NC x D
    T_minus_t = np.abs(T - t).reshape(1, NC, 1)
    sqrt_term = np.sqrt(2.0 * T_minus_t)
    X_expanded = X.reshape(1, NC, D)
    exponent = -g(X_expanded + sqrt_term * W)
    exp_mean = np.mean(np.exp(exponent), axis=0)  # Shape: (NC, 1)
    return -np.log(exp_mean[:, 0])  # Shape: (NC,)


# Main execution block
if __name__ == "__main__":
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions

    layers = [D + 1] + 4 * [256] + [1]

    Xi = np.zeros([1, D], dtype=np.float32)
    T = 1.0

    # Training
    model = HamiltonJacobiBellman(Xi, T, M, N, D, layers)

    model.train(N_Iter=50, learning_rate_schedule=learning_rate_schedule)

    # Fetch test data
    t_test, W_test = model.fetch_minibatch()

    # Make predictions
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    # Compute exact solution
    Y_test = u_exact(t_test[0, :, 0], X_pred[0, :, :])  # Shape: (NC,)

    # Compute terminal condition
    Y_test_terminal = np.log(0.5 + 0.5 * np.sum(X_pred[:, -1, :] ** 2, axis=1, keepdims=True))  # Shape: (M, 1)

    # Plot results
    plt.figure()
    plt.plot(t_test[0, :, 0], Y_pred[0, :, 0], 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0, :, 0], Y_test, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0, -1, 0], Y_test_terminal[0], 'ks', label='$Y_T = u(T,X_T)$')
    plt.plot([0], Y_test[0], 'ko', label='$Y_0 = u(0,X_0)$')
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('100-dimensional Hamilton-Jacobi-Bellman')
    plt.legend()
    plt.savefig('./figures/HJB_Sep21_50')

    plt.show()

    # Compute errors
    errors = np.sqrt((Y_test - Y_pred[0, :, 0]) ** 2 / Y_test ** 2)

    # Plot errors
    plt.figure()
    plt.plot(t_test[0, :, 0], errors, 'b')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title('100-dimensional Hamilton-Jacobi-Bellman')
    plt.savefig('./figures/HJB_Sep21_50_errors')
    plt.show()


