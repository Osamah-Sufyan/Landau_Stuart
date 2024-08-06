# import dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import itertools
import random

class LaserDESolver:
    def __init__(self, D_omeg, d_omeg, K, theta, T, T2, p, tau):
        self.D_omeg = D_omeg
        self.d_omeg = d_omeg
        self.K = K
        self.theta = theta
        self.T = T
        self.T2 = T2
        self.p = p
        self.tau = tau
        self.dt = 0.03

    def hermite_interpolation(self, p0, m0, p1, m1, t=0.5):
        # Hermite interpolation formula
        return (2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 * self.dt + (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * m1 * self.dt

    def derivatives(self, y, p0, p1, m0, m1, interpol=False, t=0.5):
        E1, E2, P1, P2, N1, N2 = y
        if interpol:
            # Interpolated values for delayed E1 and E2
            E1_delayed = self.hermite_interpolation(p0[0], m0[0], p1[0], m1[0], t)
            E2_delayed = self.hermite_interpolation(p0[1], m0[1], p1[1], m1[1], t)
        else:
            E1_delayed, E2_delayed = p0[0], p0[1]  # No interpolation needed
        
        c = ((2*self.D_omeg/(self.T2+2))**2) + 1
        dE1_dt = self.K * np.exp(1j * self.theta) * E2_delayed - 0.5 * E1 + (-1) * 1j * self.d_omeg * E1 + c*P1
        dE2_dt = self.K * np.exp(1j * self.theta) * E1_delayed - 0.5 * E2 + 1j * self.d_omeg * E2 + c * P2
        dP1_dt = (1 / self.T2) * ((1j * self.D_omeg - 1) * P1 + E1 * N1) - 1j * self.d_omeg * P1
        dP2_dt = (1 / self.T2) * ((1j * self.D_omeg - 1) * P2 + E2 * N2) + 1j * self.d_omeg * P2
        dN1_dt = (1 / self.T) * (self.p - N1 - 2 * c*np.real(P1 * np.conj(E1)))
        dN2_dt = (1 / self.T) * (self.p - N2 - 2 * c*np.real(P2 * np.conj(E2)))
        return np.array([dE1_dt, dE2_dt, dP1_dt, dP2_dt, dN1_dt, dN2_dt], dtype=complex)

    def rk4_step(self, y, delay_data, dt):
        p0, p1, m0, m1 = delay_data
        k1 = dt * self.derivatives(y, p0, p1, m0, m1)
        y_mid = y + 0.5 * k1
        k2 = dt * self.derivatives(y_mid, p0, p1, m0, m1, interpol=True, t=0.5)
        y_mid = y + 0.5 * k2
        k3 = dt * self.derivatives(y_mid, p0, p1, m0, m1, interpol=True, t=0.5)
        k4 = dt * self.derivatives(y + k3, p0, p1, m0, m1)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self):
        time = np.arange(0, 15000, self.dt)
        delay_steps = int(self.tau / self.dt)

        states = np.zeros((len(time), 6), dtype=complex)
        states[0, :4] = np.random.rand(4) + 1j * np.random.rand(4)  # Initialize E1, E2, P1, P2 with random complex numbers
        states[0, 4:] = np.random.rand(2)  # Initialize N1 and N2 with random real numbers

        for i in range(1, len(time)):
            current_index = i
            delay_index = max(0, i - delay_steps)  # Ensure positive index
            delay_data = (states[delay_index], states[max(0, delay_index - 1)],
                          (states[delay_index] - states[max(0, delay_index - 1)]) / self.dt,
                          (states[current_index] - states[max(0, current_index - 1)]) / self.dt)
            states[i] = self.rk4_step(states[i-1], delay_data, self.dt)

        E1_abs = np.abs(states[:, 0])**2
        return E1_abs, time
    
    @staticmethod
    def count_unique_maxima(x,tolerance= 1e-2):
        unique_maxima = np.array([])
        if np.all(np.isclose(x, x[0], atol=tolerance)):
            num_maxima = 0
        else:
            max_indices = np.where((np.roll(x, 1) < x) & (np.roll(x, -1) < x))[0]
            max_indices = max_indices[(max_indices != 0) & (max_indices != len(x)-1)]  # Exclude first and last index
            unique_maxima = np.unique(np.round(x[max_indices], int(np.ceil(-np.log10(tolerance)))))
        num_maxima = len(unique_maxima)     
        return unique_maxima, num_maxima
    

    def count_unique_minima(x, tolerance=1e-2):
    
        unique_minima = np.array([])
        if np.all(np.isclose(x, x[0], atol=tolerance)):
            num_minima = 0
        else:
            min_indices = np.where((np.roll(x, 1) > x) & (np.roll(x, -1) > x))[0]
            unique_minima = np.unique(np.round(x[min_indices], int(np.ceil(-np.log10(tolerance)))))
        num_minima = len(unique_minima)
        return unique_minima, num_minima

# Example of usage
def simulate(params):
    d_omeg, theta = params
    solver = LaserDESolver(D_omeg=3, d_omeg=d_omeg, K=0.1, theta=theta, T=392,T2=1, p=2, tau= 10)
    E1_abs, time = solver.solve()
    unique_maxima, num_maxima = LaserDESolver.count_unique_maxima(E1_abs[round(13000/0.03):])
    unique_minima, num_minima = LaserDESolver.count_unique_minima(E1_abs[round(13000/0.03):])


    if num_maxima > 0:
        max_intensity = max(unique_maxima)
    else:
        max_intensity = E1_abs[round(13000/0.03)]  
    
    if num_minima > 0:
        min_intensity = min(unique_minima)
    else:
        min_intensity = E1_abs[round(13000/0.03)]  
    return theta, max_intensity, min_intensity


if __name__ == '__main__':


    mesh = 20

    tolerance = 1e-0
    d_omeg = 0.0  
    params = [(d_omeg, theta) for theta in np.linspace(0, 1.3*np.pi/2, mesh)]
    with Pool() as p:
        results = p.map(simulate, params)

    
    thetas, maxima, minima = zip(*results)

    plt.figure()
    plt.plot(thetas, maxima, '-', linewidth=2, label='Maxima')
    plt.plot(thetas, minima, '-', linewidth=2, label='Minima')
    plt.title(f'$\\delta\\omega$ = 0')
    plt.xlabel(f'$\\theta$', fontsize=16)
    plt.ylabel(f'$|E_1|^2$', fontsize=16)
   
    plt.legend()
    plt.savefig('1d_bifucation_class_c.pdf')
    plt.show()