import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0. Helper functions
# ============================================================

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def onestep_predict(a1, b1, y0, u_hist, y_hist):
    """
    1-step-ahead predictor for ARX(1,1):
    y_hat[t] = a1 * y[t-1] + b1 * u[t-1]
    """
    y_hat = [y0]
    for k in range(1, len(y_hist)):
        y_hat.append(a1 * y_hist[k - 1] + b1 * u_hist[k - 1])
    return np.array(y_hat)


# ============================================================
# 1. System setup (tuned for visible bias but stable)
# ============================================================

np.random.seed(0)

# True plant parameters:
# y(t) = a1_true * y(t-1) + b1_true * u(t-1) + process_noise
a1_true = 0.9
b1_true = 0.5

# Closed-loop controller gain:
# u(t) = r(t) - K * ( y(t-1) + measurement_noise )
# -> feedback from noisy measurement
K = 2.0  # moderate, keeps closed-loop stable but couples noise into u(t)

# Noise levels
process_noise_std = 0.1   # disturbances acting on y(t)
meas_noise_std = 0.6      # sensor noise seen by controller (pretty big to induce bias)

# Number of samples
N = 2000

# ============================================================
# 2. Generate external reference r(t)
#    - slow, low-frequency signal
#    - this will be our instrument later
# ============================================================

r = np.zeros(N)
for t in range(1, N):
    # very slow drift / random walk-ish
    r[t] = 0.995 * r[t - 1] + 0.005 * np.random.randn()


# ============================================================
# 3. Simulate closed-loop system
#
# Plant:
#   y(t) = a1_true * y(t-1) + b1_true * u(t-1) + e_t
#
# Controller:
#   u(t) = r(t) - K * ( y(t-1) + n_t )
#
# Key point:
#   u(t) depends on y(t-1) + noise.
#   So u(t-1) is statistically correlated with the noise that also
#   affects y(t), which breaks the OLS assumptions.
#   => OLS will be biased.
#
# We'll identify based on (u(t), y(t)).
# ============================================================

y = np.zeros(N)
u = np.zeros(N)

for t in range(1, N):
    # noise seen by controller (measurement noise)
    n_t = meas_noise_std * np.random.randn()

    # controller acts on noisy measurement of last output
    y_meas_prev = y[t - 1] + n_t

    # control law
    u[t] = r[t] - K * y_meas_prev

    # process noise / unmodeled disturbance
    e_t = process_noise_std * np.random.randn()

    # plant update
    y[t] = (
        a1_true * y[t - 1] +
        b1_true * u[t - 1] +
        e_t
    )

# what we "observe" experimentally (output)
y_obs = y  # here we assume we log the true output y(t). You could also add measurement noise here if you want.


# ============================================================
# 4. Build regression for ARX(1,1):
#    y(t) ≈ a1 * y(t-1) + b1 * u(t-1)
# ============================================================

phi = []
Y = []
for t in range(1, N):
    phi.append([y_obs[t - 1], u[t - 1]])
    Y.append(y_obs[t])

phi = np.array(phi)           # shape (N-1, 2)
Y = np.array(Y).reshape(-1, 1)


# ============================================================
# 5. OLS estimate (biased in closed loop)
#    theta_ols = (phi^T phi)^(-1) phi^T Y
# ============================================================

theta_ols = np.linalg.inv(phi.T @ phi) @ (phi.T @ Y)
a1_ols = theta_ols[0, 0]
b1_ols = theta_ols[1, 0]


# ============================================================
# 6. IV estimate
#
# Instrument choice:
# We need something:
#  - correlated with u(t-1), y(t-1),
#  - but NOT correlated with the noise that leaked into u(t-1).
#
# The reference r(t) is perfect: it's external and noise-free.
#
# We'll build instruments Z with lagged r:
#   z1(t) = r(t-1)
#   z2(t) = r(t-2)
#
# Then:
#   theta_iv = (Z^T phi_cut)^(-1) Z^T Y_cut
#
# We have to align indices so shapes match.
# ============================================================

Z = []
for t in range(2, N):
    Z.append([r[t - 1], r[t - 2]])  # two columns -> enough to identify [a1, b1]

Z = np.array(Z)          # shape (N-2, 2)
phi_cut = phi[1:, :]     # drop first row -> align length with Z
Y_cut = Y[1:, :]         # drop first row -> align too

M = Z.T @ phi_cut
theta_iv = np.linalg.inv(M) @ (Z.T @ Y_cut)
a1_iv = theta_iv[0, 0]
b1_iv = theta_iv[1, 0]


# ============================================================
# 7. 1-step-ahead predictions for visualization
# ============================================================

y_hat_true = onestep_predict(a1_true, b1_true, y_obs[0], u, y_obs)
y_hat_ols  = onestep_predict(a1_ols,  b1_ols,  y_obs[0], u, y_obs)
y_hat_iv   = onestep_predict(a1_iv,   b1_iv,   y_obs[0], u, y_obs)

rmse_true = rmse(y_obs, y_hat_true)
rmse_ols  = rmse(y_obs, y_hat_ols)
rmse_iv   = rmse(y_obs, y_hat_iv)


# ============================================================
# 8. Print summary results
# ============================================================

print("TRUE SYSTEM:")
print(f"  a1_true = {a1_true:.4f}")
print(f"  b1_true = {b1_true:.4f}")
print()

print("OLS ESTIMATE (should be biased):")
print(f"  a1_ols  = {a1_ols:.4f}")
print(f"  b1_ols  = {b1_ols:.4f}")
print()

print("IV ESTIMATE (should be close to true):")
print(f"  a1_iv   = {a1_iv:.4f}")
print(f"  b1_iv   = {b1_iv:.4f}")
print()

print("RMSE vs observed output y(t):")
print(f"  True model RMSE : {rmse_true:.4f}")
print(f"  OLS model RMSE  : {rmse_ols:.4f}")
print(f"  IV model RMSE   : {rmse_iv:.4f}")
print()

# This is gold for writing:
bias_a1_ols = a1_ols - a1_true
bias_b1_ols = b1_ols - b1_true
bias_a1_iv  = a1_iv  - a1_true
bias_b1_iv  = b1_iv  - b1_true

print("Parameter bias (estimate - true):")
print(f"  OLS bias in a1: {bias_a1_ols:.4f}    IV bias in a1: {bias_a1_iv:.4f}")
print(f"  OLS bias in b1: {bias_b1_ols:.4f}    IV bias in b1: {bias_b1_iv:.4f}")


# ============================================================
# 9. Plot predictions and error (for slides)
# ============================================================

t0 = 200
t1 = 260
t_axis = np.arange(t0, t1)

plt.figure(figsize=(11,5))
plt.plot(t_axis, y_obs[t0:t1], label='Observed y(t)', linewidth=2)
plt.plot(t_axis, y_hat_true[t0:t1], label='True model pred', linewidth=2)
plt.plot(t_axis, y_hat_ols[t0:t1], '--', label='OLS pred', linewidth=2)
plt.plot(t_axis, y_hat_iv[t0:t1], ':', label='IV pred', linewidth=2)
plt.xlabel('Time step')
plt.ylabel('Output y')
plt.title('1-step-ahead predictions (zoomed)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Error plot (shows how OLS drifts from ground truth model)
plt.figure(figsize=(11,5))
plt.plot(t_axis,
         (y_hat_ols - y_hat_true)[t0:t1],
         '--',
         label='OLS pred error vs true model')
plt.plot(t_axis,
         (y_hat_iv - y_hat_true)[t0:t1],
         ':',
         label='IV pred error vs true model')
plt.xlabel('Time step')
plt.ylabel('Prediction error')
plt.title('Model error relative to ground truth dynamics')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
