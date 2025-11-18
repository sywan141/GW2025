# 导入库
# import sys
# import math
import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb
from astropy import units as u
import emcee
import corner
from multiprocessing import Pool, cpu_count

day = 86400.0
jetType = 2
specType = 0
z = 0.0099

t_xray, f_xray = np.loadtxt('x_ray.txt', usecols=(0, 1), unpack=True)
t_radio, f_radio = np.loadtxt('radio.txt', usecols=(0, 1), unpack=True)
t_f814w, f_f814w = np.loadtxt('f814w.txt', usecols=(0, 1), unpack=True)
t_f606w, f_f606w = np.loadtxt('f606w.txt', usecols=(0, 1), unpack=True)

nu_radio = np.full(t_radio.shape, 3.0e9)  # 3 GHz
nu_f814w = np.full(t_f814w.shape, (8369.51 * u.Angstrom).to(u.Hz, equivalencies=u.spectral()).value)
nu_f606w = np.full(t_f606w.shape, (5778.32 * u.Angstrom).to(u.Hz, equivalencies=u.spectral()).value)
nu_X = np.full(t_xray.shape, (1. * u.keV).to(u.Hz, equivalencies=u.spectral()).value)

all_time = np.concatenate((t_radio, t_f814w, t_f606w, t_xray), axis=None)
all_freq = np.concatenate((nu_radio, nu_f814w, nu_f606w, nu_X), axis=None)
x = np.row_stack([all_time, all_freq])
y = np.concatenate((f_radio, f_f814w, f_f606w, f_xray), axis=None)
sigma = 0.2 * np.abs(y)  # error (20%)

def model(x, thV, thC, log_ee, log_eb, log_E0, log_n0):
    epse = 10.**(log_ee)
    epsB = 10.**(log_eb)
    E0 = 10.0**log_E0
    n0 = 10.**(log_n0)
    if epsB <= 0 or epsB > 1 or epse <= 0 or epse > 1:
        return np.full(x.shape[1], np.nan)

    #  fixed parameters
    Z = {
        'jetType': jetType,
        'specType': specType,
        'thetaObs': thV,
        'E0': E0,
        'thetaCore': thC,
        'thetaWing': 15. / 180. * np.pi,
        'n0': n0,
        'p': 2.1,
        'epsilon_e': epse,
        'epsilon_B': epsB,
        'xi_N': 1.0,
        'd_L': 1.23e26,
        'z': z
    }

    t_day, frequency = x
    t_seconds = t_day * day
    return grb.fluxDensity(t_seconds, frequency, **Z)

def log_prior(theta):
    thV, thC, log_ee, log_eb, log_E0, log_n0 = theta
    # boundaries
    if (3.14*15/180. <= thV <= 3.14*30/180. and
        0.01 <= thC <= 0.15 and
        -4. <= log_ee <= -1. and
        -4. <= log_eb <= -1. and
        48. <= log_E0 <= 55. and
        -6. <= log_n0 <= -1.):
        return 0.0
    return -np.inf

def log_likelihood(theta, x, y, sigma):
    model_pred = model(x, *theta)
    if np.any(np.isnan(model_pred)):
        return -np.inf
    log_lkh = -0.5 * np.sum(
        (y - model_pred)**2 / sigma**2 + np.log(2 * np.pi * sigma**2)
    )
    return log_lkh

def log_probability(theta, x, y, sigma):
    lp = log_prior(theta)
    lk = log_likelihood(theta, x, y, sigma)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lk


ndim = 6  # number of free parametres
nwalkers = 32
initial_steps = 100
nsteps = 1000
optimal_processes = cpu_count()  # multiprocess

guess_par = [
    3.14*20/180.,
    0.07,
    -1.3,
    -2.4,
    52.,
    -4.
]

pos = guess_par + 1e-4 * np.random.randn(nwalkers, ndim)

if __name__ == '__main__':

    with Pool(processes=optimal_processes) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(x, y, sigma)
        )
        state = sampler.run_mcmc(pos, initial_steps, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, nsteps, progress=True)
        tau = sampler.get_autocorr_time(tol=0)
        burnin = int(5*np.max(tau))
        thin = int(0.1*np.min(tau))
        flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    proba_list = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    paras_list = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    np.savetxt('M_chi2_re.txt', -2*proba_list)  # modified chi square
    np.savetxt('Paralist_re.txt', paras_list)  # parameter sets

    best_theta = np.median(flat_samples, axis=0)
    print("\nMedians for all parameters：")
    param_names = [
        r'$\theta_V$ (rad)', r'$\theta_C$ (rad)', r'$\log \epsilon_e$',
        r'$\log \epsilon_B$', r'$\log E_0$', r'$\log n_0$'
    ]
    for name, val in zip(param_names, best_theta):
        print(f"{name}: {val:.4f}")

    fig = corner.corner(
        flat_samples,
        labels=param_names,
        truths=best_theta,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        hist_kwargs={"density": True}
    )
    plt.savefig("corner_plot_re.png", dpi=300)
    plt.close()

    t_theo = np.logspace(np.log10(1.0), np.log10(1000.0), num=100)*day
    nu_arr = [
        3.0e9,
        (8369.51 * u.Angstrom).to(u.Hz, equivalencies=u.spectral()).value,
        (5778.32 * u.Angstrom).to(u.Hz, equivalencies=u.spectral()).value,
        (1. * u.keV).to(u.Hz, equivalencies=u.spectral()).value
    ]
    labels = ['3 GHz', 'F814W', 'F606W', '1 keV']
    colors = ['blue', 'red', 'green', 'black']

    Fnu_theo = []
    for nu in nu_arr:
        x_theo = np.row_stack([t_theo / day, np.full_like(t_theo, nu)])
        Fnu = model(x_theo, *best_theta)
        Fnu_theo.append(Fnu)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_title('GRB 170817A')
    for i in range(4):
        ax.plot(t_theo / day, Fnu_theo[i], color=colors[i])

    ax.plot(t_radio, f_radio, '.', color=colors[0], alpha=0.6, label=labels[0])
    ax.plot(t_f814w, f_f814w, '.', color=colors[1], alpha=0.6, label=labels[1])
    ax.plot(t_f606w, f_f606w, '.', color=colors[2], alpha=0.6, label=labels[2])
    ax.plot(t_xray, f_xray, '.', color=colors[3], alpha=0.6, label=labels[3])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Time ($t$, d)', fontsize=12)
    ax.set_ylabel(r'Flux density ($F_\nu$, mJy)', fontsize=12)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.legend()
    plt.savefig("mcmc_fit_re.png", dpi=300)
    plt.show()