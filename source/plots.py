import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import arviz as az

from matplotlib.colors import to_rgba
from sklearn.metrics import r2_score
from source.autoencoder import AESurrogateModel
from arviz.stats.density_utils import _fast_kde_2d, \
    _find_hdi_contours
from shapely.geometry import Polygon


def calc_stat(idata):
    gridsize = (128, 128)
    density, xmin, xmax, ymin, ymax = _fast_kde_2d(idata.posterior['alpha'],
                                                   idata.posterior['tau'],
                                                   gridsize=gridsize)
    hdi_probs = [0.2]
    # Calculate contour levels and sort for matplotlib
    contour_levels = _find_hdi_contours(density, hdi_probs)
    contour_level_list = list(contour_levels) + [density.max()]
    contour_kwargs = {'levels': contour_level_list}

    g_s = complex(gridsize[0])
    x_x, y_y = np.mgrid[xmin:xmax:g_s, ymin:ymax:g_s]
    fig, ax = plt.subplots(1, 1)
    
    cs = ax.contour(x_x, y_y, density, **contour_kwargs)
    plt.close()
    
    n_different_areas = len(cs.allsegs[0])
    
    pols = [Polygon(cs.allsegs[0][i]) for i in range(n_different_areas)]
    centroid = MultiPolygon(pols).centroid
    p0_mode, p1_mode = centroid.x, centroid.y
    return p0_mode, p1_mode


def draw_bar_param(ax, idata, true_params_dict,
                   param_name: str, fontsize=14):
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    alpha_arr = posterior["alpha"].values
    beta_arr = posterior["tau"].values
    params_dict = {'alpha_arr': alpha_arr, 'beta_arr': beta_arr}
    param_arr = params_dict[param_name + '_arr']

    alpha_best, beta_best = calc_stat(idata)
    mode_value = None
    if param_name == 'alpha':
        mode_value = alpha_best
    elif param_name == 'tau':
        mode_value = beta_best

    counts, bins, patches = ax.hist(param_arr, color='RoyalBlue', alpha=1,
                                    bins=30, density=False)
    probabilities = counts / counts.sum()
    ax.clear()

    ax.bar(bins[:-1], probabilities, width=np.diff(bins),
           color='RoyalBlue', alpha=0.5, align='edge')
    # ax.hist(param_arr, color='RoyalBlue', alpha=0.2, bins=30, density=True)

    ax.vlines(true_params_dict[param_name], ymin=0, ymax=max(probabilities),
              color='White', lw=5)
    ax.vlines(true_params_dict[param_name], ymin=0, ymax=max(probabilities),
              color='OrangeRed', linestyles='dashed', label='True value', lw=3)
    ax.vlines(mode_value, ymin=0, ymax=max(probabilities), color='White',
              linestyles='solid', lw=5)
    ax.vlines(mode_value, ymin=0, ymax=max(probabilities), color='ForestGreen',
              linestyles='solid', label='Predicted value', lw=3)
    ax.set_xlabel(r'$\{}$'.format(param_name), fontsize=1.2*fontsize)
    ax.set_ylabel('Frequency', fontsize=1.2*fontsize)
    ax.set_ylim([0, 0.2])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid()
    ax.legend(fontsize=fontsize)


def draw_scatter(ax, idata, true_params_dict, fontsize=14):
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    alpha_arr = posterior["alpha"].values
    beta_arr = posterior["tau"].values
    params_dict = {'alpha_arr': alpha_arr, 'beta_arr': beta_arr}

    ax.scatter(params_dict['beta_arr'], params_dict['alpha_arr'],
               color='RoyalBlue', alpha=0.01)
    ax.scatter(params_dict['beta_arr'][-1], params_dict['alpha_arr'][-1],
               color='RoyalBlue', alpha=0.2, label='ABC parameter values')
    ax.scatter(true_params_dict['tau'], true_params_dict['alpha'],
               color='White', s=100)
    ax.scatter(true_params_dict['tau'], true_params_dict['alpha'],
               color='OrangeRed', label='True parameter values', s=50)
    alpha_best, beta_best = calc_stat(idata)

    ax.scatter(beta_best, alpha_best,
               color='White',
               s=100)
    ax.scatter(beta_best, alpha_best,
               color='ForestGreen', label='Best values',
               s=50)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1)
    ax.set_xlabel(r'$\beta_n$', fontsize=1.2*fontsize)
    ax.set_ylabel(r'$\alpha$', fontsize=1.2*fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.grid()
    # ax.set_title('Scatter Plot')


def mode_for_floats(arr):
    mode_val = stats.mode(arr.round(2))[0]
    return mode_val


def draw_lines(ax, idata, observed, true_params_dict,
               model=AESurrogateModel(10**5), cut=100, fontsize=14):

    posterior = idata.posterior.stack(samples=("draw", "chain"))
    alpha_arr = posterior["alpha"].values
    beta_arr = posterior["tau"].values
    params_dict = {'alpha_arr': alpha_arr, 'beta_arr': beta_arr}

    for index, (alpha, beta) in enumerate(zip(params_dict['alpha_arr'],
                                              params_dict['beta_arr'])):
        if index >= 1000:
            break
        ax.plot(model.simulate(alpha, beta)[
                :cut], color='RoyalBlue', alpha=0.1)
    timespace = np.arange(len(model.simulate(alpha, beta)))
    ax.plot(model.simulate(alpha, beta)[
            :cut], color='RoyalBlue', alpha=0.8, label='Simulation')

    # observed = model.simulate(
    #     true_params_dict['alpha'], true_params_dict['beta'])[:cut]
    ax.scatter(timespace[:cut], observed[:cut],
               color='White', s=20, zorder=10001)
    ax.scatter(timespace[:cut], observed[:cut],
               color='OrangeRed', s=10, label='Observed incidence', zorder=10002)

    alpha_best, beta_best = calc_stat(idata)
    best_simulation = model.simulate(alpha_best, beta_best)[:cut]

    ax.plot(best_simulation,
            color='White', lw=4,
            zorder=9999)
    r2_best = round(
        r2_score(best_simulation[:cut], observed[:cut]), 2)
    ax.plot(best_simulation,
            color='ForestGreen', lw=2,
            label=r'Best simulation ($R^2={}$)'.format(r2_best),
            zorder=9999)
    # ax.text(75, 1500, r'$R^2={}$'.format(round(
    #     r2_score(best_simulation, true_incidence), 2)), fontsize=fontsize, color='k')

    ax.set_xlabel('Time, days', fontsize=1.2*fontsize)
    ax.set_ylabel('Incidence', fontsize=1.2*fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid()
    ax.legend(fontsize=fontsize)


def create_plots_grid(idata, observed, true_params_dict, n_rows=2, n_cols=2,
                      plot_funcs=[draw_scatter, draw_lines,
                                  draw_bar_param,  draw_bar_param], fontsize=14):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    labels = [alphabet[index] + ')' for index in range(len(alphabet))]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    total_cells = n_rows * n_cols

    for idx in range(total_cells):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        plot_func = plot_funcs[idx % len(plot_funcs)]
        if idx == 2:
            plot_func(ax, idata, true_params_dict, param_name='alpha')
        elif idx == 3:
            plot_func(ax, idata, true_params_dict, param_name='beta')
        else:
            plot_func(ax, idata, true_params_dict)

        ax.annotate(labels[idx], xy=(0, 0), xycoords='axes fraction',
                    xytext=(-30, -30), textcoords='offset points',
                    fontsize=1.5*fontsize, ha='right', va='baseline')
    plt.tight_layout()
    return fig


'''
def plot_calib(idata, observed_incidence, true_params_dict, fontsize=14):
    cmap = mpl.colormaps['viridis']
    hdi_list = [0.1, 0.2, 0.5, 0.8, 0.9]
    colors_l = cmap(np.linspace(0, 1, len(hdi_list)))

    # for edgecolor to have alpha
    fc = to_rgba('RoyalBlue', 0.5)
    param_names = ['beta', 'alpha']
    fancy_names = [r'$\beta_n$', r'$\alpha$']

    fig = plt.figure(figsize=(12, 5))
    # adding gridspec
    gs = fig.add_gridspec(1, 2, hspace=0.6, width_ratios=[1, 1.25])
    # dividing it even further!
    gs0 = gs[0].subgridspec(2, 1, height_ratios=[1, 4], hspace=0.)
    gs1 = gs[1].subgridspec(2, 2, wspace=0, hspace=0.,
                            width_ratios=[5, 1],
                            height_ratios=[1, 4])

    # creating subplots, ignoring useless corners
    ax_curves = fig.add_subplot(gs0[1])

    ax_up = fig.add_subplot(gs1[0])
    ax_scatter = fig.add_subplot(gs1[2], sharex=ax_up)
    ax_right = fig.add_subplot(gs1[3], sharey=ax_scatter)

    # plotting UFO
    az.plot_pair(
        idata,
        var_names=["beta", "alpha"],
        kind=["scatter", "kde"],
        kde_kwargs={"fill_last": False,
                    'hdi_probs': hdi_list,
                    'fill_kwargs': {'alpha': .1},
                    'contour_kwargs': {"colors": None},
                    'contourf_kwargs': {"alpha": 0}},
        marginals=True,
        marginal_kwargs={'kind': 'hist', 'hist_kwargs': {'bins': 50,
                                                         'color': fc,
                                                         # 'alpha':.5,
                                                         'ec': fc}},
        scatter_kwargs={'color': fc},
        ax=np.array([[ax_up, None], [ax_scatter, ax_right]])
    )

    # removing ticks from small plots
    for a in [ax_up, ax_right]:
        plt.setp(a.get_xticklabels(), visible=False)
        plt.setp(a.get_yticklabels(), visible=False)
    # setting normal ticks for a scatterplot
    min_x = idata.posterior[param_names[0]].min().round(1)
    min_y = idata.posterior[param_names[1]].min().round(1)

    ax_scatter.set_xticks(np.arange(min_x, 1, 0.2),
                          np.arange(min_x, 1, 0.2).round(1),
                          fontsize=10)
    ax_scatter.set_yticks(np.arange(min_y, 1, 0.1),
                          np.arange(min_y, 1, 0.1).round(1),
                          fontsize=10)

    p0_mode, p1_mode = calc_stat(idata)

    # i don't know if there can be multiple ref points, so it's easier
    ls1 = ax_scatter.scatter(true_params_dict['beta'], true_params_dict['alpha'],
                             alpha=0.9,
                             color='OrangeRed', label='Observed',
                             edgecolors='white',
                             s=50, zorder=99)

    ls2 = ax_scatter.scatter(p1_mode, p0_mode, alpha=0.9,
                             color='tab:green', label='Selected',
                             edgecolors='white',
                             s=50, zorder=99)
    ls3 = ax_scatter.scatter(true_params_dict['beta'], true_params_dict['alpha'],
                             zorder=0, s=5, color=fc, label='Simulation')

    ax_scatter.tick_params(axis='both', which='major', labelsize=fontsize)
    ax_scatter.set_xlabel(fancy_names[0], fontsize=1.2*fontsize)
    ax_scatter.set_ylabel(fancy_names[1], fontsize=1.2*fontsize)

    legend_elements = []
    for c, val in zip(colors_l[::-1], hdi_list):
        legend_elements.append(mpatches.Patch(color=c,
                                              label=f'HDR {int(val*100)}%',
                                              alpha=.9)
                               )

    ax_scatter.legend(handles=[ls1, ls2, ls3,
                               *legend_elements])
    ax_scatter.grid()

    ax_up.axvline(p1_mode, ls='-', color='white', lw=3)
    ax_up.axvline(p1_mode, ls='--', color='tab:green', label='Selected')
    ax_up.axvline(true_params_dict['beta'], ls='-', color='white', lw=3)
    ax_up.axvline(true_params_dict['beta'], ls='--',
                  color='OrangeRed', label='Observed')
    ax_up.legend()

    ax_right.axhline(p0_mode, ls='-', color='white', lw=3)
    ax_right.axhline(p0_mode, ls='--', color='tab:green', label='Selected')
    ax_right.axhline(true_params_dict['alpha'], ls='-', color='white', lw=3)
    ax_right.axhline(true_params_dict['alpha'],
                     ls='--', color='OrangeRed', label='Observed')

    draw_lines(ax=ax_curves, idata=idata, observed=observed_incidence,
               true_params_dict=true_params_dict)
    ax_curves.legend(fontsize=10)
    plt.tight_layout()
    return fig
'''


def draw_forecast(ax, idata, observed, full_data,
                  forecast_period=7,
                  model=AESurrogateModel(10**5),
                  fontsize=14, cut=100):
    posterior = idata.posterior.stack(samples=("draw", "chain"))
    alpha_arr = posterior["alpha"].values
    beta_arr = posterior["beta"].values
    params_dict = {'alpha_arr': alpha_arr, 'beta_arr': beta_arr}
    timespace = np.arange(len(full_data))
    for index, (alpha, beta) in enumerate(zip(params_dict['alpha_arr'], params_dict['beta_arr'])):
        if index >= 1000:
            break
        ax.plot(timespace[len(observed):len(observed) + forecast_period][:cut],
                model.simulate(alpha, beta)[len(observed):len(
                    observed) + forecast_period][:cut],
                color='RoyalBlue', alpha=0.05)

    ax.plot(timespace[len(observed):len(observed) + forecast_period][:cut],
            model.simulate(alpha, beta)[len(observed):len(
                observed) + forecast_period][:cut],
            color='RoyalBlue',
            alpha=0.8, label='Forecast')

    ax.scatter(timespace[:len(observed)][:cut], observed[:cut], s=50, color='LimeGreen', edgecolors='k',
               label='Known data', zorder=10000)
    ax.scatter(timespace[len(observed):][:cut-len(observed)], full_data[len(observed):][:cut-len(observed)], s=50, color='Grey', edgecolors='k',
               label='Unknown data', zorder=10001, alpha=0.5)

    ax.set_xlabel('Time, days', fontsize=1.2*fontsize)
    ax.set_ylabel('Incidence', fontsize=1.2*fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ymax = 3300
    ax.vlines(len(observed), ymin=0, ymax=ymax, color='k', linestyle='dashed')
    ax.set_ylim(0, ymax)
    ax.grid()
    ax.legend(fontsize=fontsize)


def plot_calib(idata, observed_incidence, true_params_dict, full_data=None, fontsize=14,
               ax_curves=[], ax_kde=[], pred=False):
    cmap = mpl.colormaps['viridis']
    hdi_list = [0.2, 0.5, 0.8, 0.9]
    colors_l = cmap(np.linspace(0, 1, len(hdi_list)))

    # for edgecolor to have alpha
    fc = to_rgba('RoyalBlue', 0.5)
    param_names = ['beta', 'alpha']
    fancy_names = [r'$\beta_n$', r'$\alpha$']

    if not ax_curves:
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(1, 2, hspace=0.6, width_ratios=[1, 1.25])
        # dividing it even further!
        gs0 = gs[0].subgridspec(2, 1, height_ratios=[1, 4], hspace=0.)
        gs1 = gs[1].subgridspec(2, 2, wspace=0, hspace=0.,
                                width_ratios=[5, 1],
                                height_ratios=[1, 4])

        # creating subplots, ignoring useless corners
        ax_curves = fig.add_subplot(gs0[1])

        ax_up = fig.add_subplot(gs1[0])
        ax_scatter = fig.add_subplot(gs1[2], sharex=ax_up)
        ax_right = fig.add_subplot(gs1[3], sharey=ax_scatter)
    else:
        ax_curves = ax_curves[0]
        ax_up, ax_scatter, ax_right = ax_kde

    # plotting UFO
    q = az.plot_pair(
        idata,
        var_names=["beta", "alpha"],
        kind=["scatter", "kde"],
        kde_kwargs={"fill_last": False,
                    'hdi_probs': hdi_list,
                    'fill_kwargs': {'alpha': .1},
                    'contour_kwargs': {"colors": None},
                    'contourf_kwargs': {"alpha": 0.3,
                                        'colors': [colors_l[-1],
                                                   *colors_l[:-1]]
                                        }},
        marginals=True,
        marginal_kwargs={'kind': 'hist', 'hist_kwargs': {'bins': 50,
                                                         'color': fc,
                                                         # 'alpha':.5,
                                                         'ec': fc}},
        scatter_kwargs={'color': fc},
        ax=np.array([[ax_up, None], [ax_scatter, ax_right]])
    )
    # removing ticks from small plots
    for a in [ax_up, ax_right]:
        plt.setp(a.get_xticklabels(), visible=False)
        plt.setp(a.get_yticklabels(), visible=False)
    # setting normal ticks for a scatterplot
    min_x = idata.posterior[param_names[0]].min().round(1)
    min_y = idata.posterior[param_names[1]].min().round(1)

    ax_scatter.set_xticks(np.arange(min_x, 1, 0.2),
                          np.arange(min_x, 1, 0.2).round(1),
                          # fontsize=10
                          )
    ax_scatter.set_yticks(np.arange(min_y, 1, 0.1),
                          np.arange(min_y, 1, 0.1).round(1),
                          # fontsize=10
                          )

    p0_mode, p1_mode = calc_stat(idata)

    ls1 = ax_scatter.scatter(true_params_dict['tau'],
                             true_params_dict['alpha'], alpha=0.9,
                             color='OrangeRed', label='Observed',
                             edgecolors='white',
                             s=50, zorder=99)

    ls2 = ax_scatter.scatter(p1_mode, p0_mode, alpha=0.9,
                             color='tab:green', label='Selected',
                             edgecolors='white',
                             s=50, zorder=99)
    ls3 = ax_scatter.scatter(true_params_dict['tau'],
                             true_params_dict['alpha'], zorder=0, s=5,
                             color=fc, label='Simulation')

    ax_scatter.set_xlabel(fancy_names[0])
    ax_scatter.set_ylabel(fancy_names[1])
    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0.5, 1)
    ticks_x = [0.1, 0.3, 0.5, 0.7, 0.9]
    ticks_y = [0.5, 0.6, 0.7, 0.8, 0.9]
    ax_scatter.set_xticks(ticks_x, list(map(str, ticks_x)))
    ax_scatter.set_yticks(ticks_y, list(map(str, ticks_y)))
    ax_scatter.tick_params(axis='both', which='major', labelsize=fontsize)
    ax_scatter.set_xlabel(fancy_names[0], fontsize=1.2*fontsize)
    ax_scatter.set_ylabel(fancy_names[1], fontsize=1.2*fontsize)

    legend_elements = []
    for c, val in zip(colors_l[::-1], hdi_list):
        legend_elements.append(mpatches.Patch(color=c,
                                              label=f'HDR {int(val*100)}%',
                                              alpha=.9))

    ax_scatter.legend(handles=[ls1, ls2, ls3,
                               *legend_elements],
                      fontsize=10)
    ax_scatter.grid()
    ax_up.axvline(p1_mode, ls='-', color='white', lw=3)
    ax_up.axvline(p1_mode, ls='--',
                  color='tab:green', label='Selected')
    ax_up.axvline(true_params_dict['tau'], ls='-', color='white', lw=3)
    ax_up.axvline(true_params_dict['tau'], ls='--',
                  color='OrangeRed', label='Observed')
    ax_up.legend()

    ax_right.axhline(p0_mode, ls='-', color='white', lw=3)
    ax_right.axhline(p0_mode, ls='--', color='tab:green')
    ax_right.axhline(true_params_dict['alpha'], ls='-', color='white', lw=3)
    ax_right.axhline(true_params_dict['alpha'], ls='--', color='OrangeRed')
    if pred:
        draw_forecast(ax=ax_curves, idata=idata, observed=observed_incidence,
                      full_data=full_data)
    else:
        draw_lines(ax=ax_curves, idata=idata, observed=observed_incidence,
                   true_params_dict=true_params_dict)
    ax_curves.legend(fontsize=10)
    if not ax_curves:
        return fig
    return


def plot_calib_subplots(idata_arr, observed_incidence_arr, true_params_dict,
                        full_data, fontsize=14):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    labels = [alphabet[index] + ')' for index in range(len(alphabet))]

    fig = plt.figure(figsize=(20, 10))

    # adding gridspec
    gs = fig.add_gridspec(2, 3, hspace=0.2, width_ratios=[1, 1, 1],
                          height_ratios=[1, 1.25])

    # curves subplots
    gs00 = gs[0, 0].subgridspec(1, 2, width_ratios=[4, 1],
                                wspace=0.)
    gs01 = gs[0, 1].subgridspec(1, 2, width_ratios=[4, 1],
                                wspace=0.)
    gs02 = gs[0, 2].subgridspec(1, 2, width_ratios=[4, 1],
                                wspace=0.)

    # ufo subplots
    gs10 = gs[1, 0].subgridspec(2, 2, wspace=0, hspace=0.,
                                width_ratios=[4, 1],
                                height_ratios=[1, 4])
    gs11 = gs[1, 1].subgridspec(2, 2, wspace=0, hspace=0.,
                                width_ratios=[4, 1],
                                height_ratios=[1, 4])
    gs12 = gs[1, 2].subgridspec(2, 2, wspace=0, hspace=0.,
                                width_ratios=[4, 1],
                                height_ratios=[1, 4])

    for gs_0i, gs_1i, idata_i, observed_incidence, idx in zip([gs00, gs01, gs02],
                                                              [gs10, gs11, gs12],
                                                              idata_arr,
                                                              observed_incidence_arr,
                                                              [i for i in range(3)]):

        ax_curves = fig.add_subplot(gs_0i[0])
        ax_up = fig.add_subplot(gs_1i[0])
        ax_scatter = fig.add_subplot(gs_1i[2], sharex=ax_up)
        ax_right = fig.add_subplot(gs_1i[3], sharey=ax_scatter)

        plot_calib(idata_i, observed_incidence,
                   true_params_dict,
                   full_data=full_data,
                   ax_curves=[ax_curves],
                   ax_kde=[ax_up, ax_scatter, ax_right],
                   pred=True)
        ax_curves.annotate(labels[idx], xy=(0, 0), xycoords='axes fraction',
                           xytext=(-30, -50), textcoords='offset points',
                           fontsize=1.5*fontsize, ha='right', va='baseline')

    return fig
