import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
import matplotlib.patches as mpatches
from scipy.stats import binned_statistic_2d
import seaborn as sns
import math
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

colour_theme = np.array(['lightsalmon', 'powderblue', 'gold', 'springgreen', 'magenta', 'plum'])


def createPlayerRegions(type):
    if type == 'rally':
        receiver_regions = {'Deuce': [272, 273, 274, 284, 285, 286, 296, 297, 298, 308, 309, 310],
                            'Middle': [270, 271, 282, 283, 294, 295, 306, 307],
                            'Ad': [267, 268, 269, 279, 280, 281, 291, 292, 293, 303, 304, 305]}
        impact_regions = {'Ad': [8, 9, 10, 20, 21, 22, 32, 33, 34, 44, 45, 46],
                          'Middle': [6, 7, 18, 19, 30, 31, 42, 43],
                          'Deuce': [3, 4, 5, 15, 16, 17, 27, 28, 29, 39, 40, 41]}
        return [impact_regions, receiver_regions]
    elif type == 'deuce_serve':
        receiver_regions = {'T': [271, 283, 295, 307],
                            'Middle': [272, 284, 296, 308],
                            'Corner': [273, 285, 297, 309]}
        return receiver_regions
    elif type == 'ad_serve':
        receiver_regions = {'T': [270, 282, 294, 306],
                            'Middle': [269, 281, 293, 305],
                            'Corner': [268, 280, 292, 304]}
        return receiver_regions


def createBins(compressed=False, only_half=False, big_buckets=False):
    if compressed == False:
        if only_half == False:
            xBin = np.asarray(
                [-18.8872, -17.8872, -16.8872, -15.8872, -14.8872, -13.8872, -12.8872, -11.8872, -10.78992, -9.69264,
                 -8.59536, -7.49808, -6.4008, -5.334, -4.2672, -3.2004, -2.1336, -1.0668, 0.0000, 1.0668, 2.1336,
                 3.2004, 4.2672, 5.334, 6.4008, 7.49808, 8.59536, 9.69264, 10.78992, 11.8872, 12.8872, 13.8872, 14.8872,
                 15.8872, 16.8872, 17.8872, 18.8872])
            yBin = np.asarray(
                [-9.4864, -8.4864, -7.4864, -6.4864, -5.4864, -823 / 200, -823 / 300, -823 / 600, 0.0000, 823 / 600,
                 823 / 300, 823 / 200, 5.4864, 6.4864, 7.4864, 8.4864, 9.4864])
        elif only_half == True:
            xBin = np.asarray(
                [-1.0668, 0.0000, 1.0668, 2.1336, 3.2004, 4.2672, 5.334, 6.4008, 7.49808,
                 8.59536, 9.69264, 10.78992, 11.8872, 12.8872, 13.8872, 14.8872, 15.8872,
                 16.8872, 17.8872, 18.8872])
            yBin = np.asarray(
                [-9.4864, -8.4864, -7.4864, -6.4864, -5.4864, -823 / 200, -823 / 300, -823 / 600, 0.0000, 823 / 600,
                 823 / 300, 823 / 200, 5.4864, 6.4864, 7.4864, 8.4864, 9.4864])
    elif compressed == True:
        if only_half == False:
            xBin = np.asarray(
                [-13.8872, -12.8872, -11.8872, -10.78992, -9.69264, -8.59536, -7.49808, -6.4008,
                 -5.334, -4.2672, -3.2004, -2.1336, -1.0668, 0.0000, 1.0668, 2.1336, 3.2004, 4.2672, 5.334, 6.4008,
                 7.49808,
                 8.59536, 9.69264, 10.78992, 11.8872, 12.8872, 13.8872])
            yBin = np.asarray(
                [-7.4864, -6.4864, -5.4864, -823 / 200, -823 / 300, -823 / 600, 0.0000, 823 / 600, 823 / 300,
                 823 / 200, 5.4864, 6.4864, 7.4864])
        elif only_half == True:
            xBin = np.asarray(
                [-1.0668, 0.0000, 1.0668, 2.1336, 3.2004, 4.2672, 5.334, 6.4008, 7.49808, 8.59536, 9.69264, 10.78992,
                 11.8872, 12.8872, 13.8872])
            yBin = np.asarray(
                [-7.4864, -6.4864, -5.4864, -823 / 200, -823 / 300, -823 / 600, 0.0000, 823 / 600, 823 / 300, 823 / 200,
                 5.4864, 6.4864, 7.4864])

    if big_buckets == True:
        if only_half == False:
            xBin = np.asarray(
                [-15.8872, -13.8872, -11.8872, -9.1414, -6.4008, -4.2672, -2.1336, 0.0000, 2.1336,
                 4.2672, 6.4008, 9.1414, 11.8872, 13.8872,
                 15.8872])
            yBin = np.asarray(
                [-7, -823 / 200, -823 / 400, 0.0000, 823 / 400, 823 / 200, 7])
        elif only_half == True:
            xBin = np.asarray(
                [0.0000, 2.1336, 4.2672, 6.4008, 9.1414, 11.8872, 13.8872, 15.8872])
            yBin = np.asarray(
                [-7, -823 / 200, -823 / 400, 0.0000, 823 / 400, 823 / 200, 7])

    return xBin, yBin


def createLabeledCourt(compressed=False, only_half=False, big_buckets=False, half_label=False):
    xBin, yBin = createBins(compressed, only_half, big_buckets)
    xBinList = xBin.tolist()
    yBinList = yBin.tolist()

    createCourt(plt.gca(), compressed, only_half, big_buckets)

    for j in yBin[0:len(yBin) - 1]:
        for i in xBin[0:len(xBin) - 1]:
            # value = (xBinList.index(i) + 1) + (yBinList.index(j)) * (len(xBin) - 1)
            value = (xBinList.index(i)) * (len(yBin) - 1) + (yBinList.index(j) + 1)  # numbers vertical

            if half_label == True:
                if value > ((len(xBin) - 1) * (len(yBin) - 1)) / 2:
                    break

            if big_buckets == False:
                plt.text(i + 0.5, j + 0.55, '%d' % value, ha='center', va='center', )
            else:
                plt.text(i + 1, j + 1, '%d' % value, ha='center', va='center', weight="bold")


def createCourt(ax=None, compressed=False, only_half=False, big_buckets=False):
    if ax == None:
        ax = plt.gca()

    if only_half == False:
        # Plot Line Markings
        plt.plot([0, 0], [-6, 6], color="white", linewidth=6.0)  # net/middle line
        plt.plot([-6.4008, 6.4008], [0, 0], color="white")  # center line
        plt.plot([-6.4008, -6.4008], [-4.115, 4.115], color="white");
        plt.plot([6.4008, 6.4008], [-4.115, 4.115], color="white")  # service lines
        plt.plot([-11.8872, -11.5], [0, 0], color="white");
        plt.plot([11.8872, 11.5], [0, 0], color="white")  # center hashmark
        plt.plot([-11.8872, 11.8872], [5.4864, 5.4864], color="white");
        plt.plot([-11.8872, 11.8872], [-5.4864, -5.4864], color="white")  # Doubles side lines
        plt.plot([-11.8872, 11.8872], [4.115, 4.115], color="white");
        plt.plot([-11.8872, 11.8872], [-4.115, -4.115], color="white")  # Singles side lines
        plt.plot([-11.8872, -11.8872], [-5.4864, 5.4864], color="white");
        plt.plot([11.8872, 11.8872], [-5.4864, 5.4864], color="white")  # base lines

        # Add Colours
        rectCourt = mpatches.Rectangle((-11.8872, -5.4864), 11.8872 * 2, 5.4864 * 2, angle=0, color='#3C638E', zorder=1)
        ax.add_patch(rectCourt)
        ax.set_facecolor('#6C935C')

    elif only_half == True:
        # Plot Line Markings
        plt.plot([0, 0], [-6, 6], color="white", linewidth=6.0)  # net/middle line
        plt.plot([0, 6.4008], [0, 0], color="white")  # center line
        plt.plot([6.4008, 6.4008], [-4.115, 4.115], color="white")  # service lines
        plt.plot([11.8872, 11.5], [0, 0], color="white")  # center hashmark
        plt.plot([0, 11.8872], [5.4864, 5.4864], color="white");
        plt.plot([0, 11.8872], [-5.4864, -5.4864], color="white")  # Doubles side lines
        plt.plot([0, 11.8872], [4.115, 4.115], color="white");
        plt.plot([0, 11.8872], [-4.115, -4.115], color="white")  # Singles side lines
        plt.plot([11.8872, 11.8872], [-5.4864, 5.4864], color="white")  # base lines

        # Add Colours
        rectCourt = mpatches.Rectangle((0, -5.4864), 11.8872, 5.4864 * 2, angle=0, color='#3C638E', zorder=1)
        ax.add_patch(rectCourt)
        ax.set_facecolor('#6C935C')

    xBin, yBin = createBins(compressed, only_half, big_buckets)
    plt.xticks(xBin)
    plt.yticks(yBin)
    plt.grid(True, linestyle='dotted', zorder=4)


def markupCourt(player, buckets, compressed=False, only_half=False, big_buckets=False):
    # Mark up the court
    ax = plt.gca()
    xBin, yBin = createBins(compressed, only_half, big_buckets)

    if player == 'impact':
        colour = "red"
    elif player == 'receiver_start':
        colour = "purple"
    elif player == 'receiver_end':
        colour = 'orange'

    for bucket in buckets:
        # x = (bucket % (len(xBin) - 1)) - 1
        # y = math.floor(bucket / (len(xBin) - 1))

        # labelling court vertically
        y = (bucket % (len(yBin) - 1)) - 1
        x = math.ceil((bucket / (len(yBin) - 1)) - 1)

        if y == -1:
            y = 5

        w = np.abs(xBin[x] - xBin[x + 1])
        h = np.abs(yBin[y] - yBin[y + 1])

        sq = mpatches.Rectangle((xBin[x], yBin[y]), w, h, angle=0, color=colour, zorder=1)
        ax.add_patch(sq)


def markupImpactPlayer(impact_region):
    ax = plt.gca()
    if impact_region == 'Deuce':
        sq = mpatches.Rectangle((-1.0668, -5.4864), 1.0668, (5.4864 - 823 / 600), angle=0, color='red', zorder=1)
    if impact_region == 'Middle':
        sq = mpatches.Rectangle((-1.0668, -823 / 600), 1.0668, (823 / 300), angle=0, color='red', zorder=1)
    if impact_region == 'Ad':
        sq = mpatches.Rectangle((-1.0668, 823 / 600), 1.0668, (5.4864 - 823 / 600), angle=0, color='red', zorder=1)
    ax.add_patch(sq)


def createMRPheatmap(V_full, count_full, bucket, choice='rally', vmin=-1, compressed=False, only_half=False, big_buckets=False,
                     half_label=False):

    xBin, yBin = createBins(compressed=True, only_half=True, big_buckets=True)

    V_bucket = V_full[choice][bucket - 1, :]
    V_bucket = np.reshape(V_bucket, (7, 6))

    count_bucket = count_full[choice][bucket - 1, :]
    count_bucket = np.reshape(count_bucket, (7, 6))

    heatmap = plt.pcolormesh(xBin, yBin, V_bucket.T, zorder=2, cmap='RdYlGn',
                             vmax=1, vmin=vmin, alpha=0.95)
    xBinList = xBin.tolist()
    yBinList = yBin.tolist()

    for j in yBin[0:len(yBin) - 1]:
        for i in xBin[0:len(xBin) - 1]:
            value = V_bucket.T[yBinList.index(j), xBinList.index(i)]
            plt.text(i + 1, j + 1.5, '%.2f' % value, ha='center', va='center', fontsize=10, weight='bold')
            value = count_bucket.T[yBinList.index(j), xBinList.index(i)]
            plt.text(i + 1, j + 1, '(%d)' % value, ha='center', va='center', fontsize=10)

    createLabeledCourt(compressed, only_half, big_buckets, half_label)
    markupCourt('impact', [bucket], compressed=True, only_half=False, big_buckets=True)


def createMDPheatmap(V_full, count_full, opt_policy_full, bucket, choice='rally', vmin=-1, compressed=False, only_half=False, big_buckets=False,
                     half_label=False, diff=False):

    xBin, yBin = createBins(compressed=True, only_half=True, big_buckets=True)

    V_bucket = V_full[choice][bucket - 1, :]
    V_bucket = np.reshape(V_bucket, (7, 6))

    count_bucket = count_full[choice][bucket - 1, :]
    count_bucket = np.reshape(count_bucket, (7, 6))

    opt_policy_bucket = opt_policy_full[choice][bucket - 1, :]
    opt_policy_bucket = np.reshape(opt_policy_bucket, (7, 6))

    heatmap = plt.pcolormesh(xBin, yBin, V_bucket.T, zorder=2, cmap='RdYlGn',
                             vmax=1, vmin=vmin, alpha=0.95)
    xBinList = xBin.tolist()
    yBinList = yBin.tolist()

    actions_dict = {0: "corner", 1: "middle", 2: "short_ad", 3: "short_mid", 4: "short_deuce", 5: "deep_ad", 6: "deep_mid", 7: "deep_deuce"}

    for j in yBin[0:len(yBin) - 1]:
        for i in xBin[0:len(xBin) - 1]:
            value = V_bucket.T[yBinList.index(j), xBinList.index(i)]
            plt.text(i + 1, j + 1.5, '%.2f' % value, ha='center', va='center', fontsize=14, weight='bold')
            if not diff:
                value = count_bucket.T[yBinList.index(j), xBinList.index(i)]
                plt.text(i + 1, j + 1, '(%d)' % value, ha='center', va='center', fontsize=14)
                value = opt_policy_bucket.T[yBinList.index(j), xBinList.index(i)]
                plt.text(i + 1, j + 0.5, '%s' % actions_dict[value], ha='center', va='center', fontsize=12)

    createLabeledCourt(compressed, only_half, big_buckets, half_label)
    markupCourt('impact', [bucket], compressed=True, only_half=False, big_buckets=True)


def createHeatmap(x, y, z, stat_name, minimum=1, annot=False, addCount=True, compressed=False, only_half=False, alpha=1,
                  vast_min=0, title="", big_buckets=False):
    # Create bins of the court
    xBin, yBin = createBins(compressed, only_half, big_buckets)
    bins = [xBin, yBin]

    # determine the means/count for each bin
    means, xedges, yedges, binnumber = binned_statistic_2d(x=x, y=y, values=z, statistic='mean', bins=bins)
    counts, xedges, yedges, binnumber = binned_statistic_2d(x=x, y=y, values=z, statistic='count', bins=bins)

    # means[means == 0] = 'nan'
    # remove squares with not enough observations
    means[counts < minimum] = 'nan'
    counts[counts < minimum] = 'nan'

    # Create heat map based on means
    heatmap = plt.pcolormesh(xBin, yBin, means.T, zorder=2, cmap='Reds', vmax=np.nanmax(means), vmin=vast_min,
                             alpha=alpha)

    xBinList = xBin.tolist()
    yBinList = yBin.tolist()

    # Annotate heatmap if desired with mean values and (count) values
    if (annot == True) or (addCount == True):
        for j in yBin[0:len(yBin) - 1]:
            for i in xBin[0:len(xBin) - 1]:
                if annot == True:
                    value = means.T[yBinList.index(j), xBinList.index(i)]
                    if not math.isnan(value):
                        plt.text(i + 0.5, j + 0.85, '%.2f' % value, ha='center', va='center', fontsize=6, weight='bold')
                if addCount == True:
                    value = counts.T[yBinList.index(j), xBinList.index(i)]
                    if not math.isnan(value):
                        plt.text(i + 0.5, j + 0.35, '(%d)' % value, ha='center', va='center', fontsize=6)

    # plt.colorbar(heatmap, label=stat_name + ' Value')
    plt.title(title)
    plt.subplots_adjust(left=0.055, bottom=0.11, right=0.95, top=0.88)
    createCourt(plt.gca(), compressed, only_half, big_buckets)

    return means


def createKde(x, y, compressed=False, only_half=False, title="", ax=None,cmap='Blues', n_levels=10, alpha=1, big_buckets=False):
    if ax == None:
        ax = plt.gca()
    sns.kdeplot(x, y, cmap=cmap, shade_lowest=False, shade="True", n_levels=n_levels, zorder=3, ax=ax, alpha=alpha)
    ax.set_ylabel('')
    ax.set_xlabel('')
    createCourt(ax, compressed, only_half, big_buckets)
    plt.title(title)


def fitGMM(df, n_components, title, compressed=False, only_half=False, big_buckets=False):
    X = df[['x02', 'y02']]
    X = X.to_numpy()

    x, y = np.meshgrid(np.sort(X[:, 0]), np.sort(X[:, 1]))
    XY = np.array([x.flatten(), y.flatten()]).T

    GMM = GaussianMixture(n_components=n_components, covariance_type='full').fit(X)  # Instantiate and fit the model
    print('Converged:', GMM.converged_)  # Check if the model has converged
    means = GMM.means_
    covariances = GMM.covariances_

    # Predict
    # Y = np.array([[0.5], [0.5]])
    prediction = GMM.predict_proba(X)
    prediction_mean = np.mean(prediction, axis=0)
    # print(prediction)

    # Plot
    plt.scatter(X[:, 0], X[:, 1], c='blue', zorder=2)
    i = 0
    for m, c in zip(means, covariances):
        multi_normal = multivariate_normal(mean=m, cov=c)
        plt.contour(np.sort(X[:, 0]), np.sort(X[:, 1]), multi_normal.pdf(XY).reshape(len(X), len(X)),
                    colors='black', alpha=0.3, zorder=2)
        plt.scatter(m[0], m[1], c='grey', zorder=10, s=100)
        plt.text(m[0] + 0.5, m[1] + 0.5, '(%0.2f)' % prediction_mean[i], ha='center', va='center', fontsize=12)
        plt.title(title)
        i = i + 1
    createCourt(plt.gca(), compressed, only_half, big_buckets)

    return means, covariances, prediction


def caption_cluster_plots(df, centroids, n_clusters):
    ax = plt.gca()
    subtitle = ''
    counts = df.groupby(['Cluster', 'error'])['vast'].count()

    for n in range(n_clusters):
        centroid_shots = df[(df.Cluster == n) & (df.x02 > centroids[n, 0] - 0.5) & (df.x02 < centroids[n, 0] + 0.5)
                            & (df.y02 > centroids[n, 1] - 0.5) & (df.y02 < centroids[n, 1] + 0.5)]

        # Make sure that the index exists
        if (n, False) in counts.index:
            count_false = counts[n, False]
        else:
            count_false = 0
        if (n, True) in counts.index:
            count_true = counts[n, True]
        else:
            count_true = 0

        subtitle = subtitle + '{} : {} (in-play) | {} (error) | vast_centroid = {:0.2f} \n'.format(colour_theme[n],
                                                                                                   count_false,
                                                                                                   count_true,
                                                                                                   centroid_shots[
                                                                                                       "vast"].mean())

    ax.set_xlabel(subtitle)


def plotCentroids(centroids, means_tmp=None, just_ball=False):
    if just_ball == True:
        for i in range(centroids.shape[0]):
            plt.scatter(centroids[i, 0], centroids[i, 1], c='black', s=60, marker="x", zorder=2, linewidth=3)
            plt.text(centroids[i, 0] + 0.75, centroids[i, 1] + 0.50, '(%0.2f)' % means_tmp[i], ha='center', va='center',
                     fontsize=9)
    else:
        for i in range(centroids.shape[0]):
            for j in range(0, centroids.shape[1], 2):
                plt.scatter(centroids[i, j], centroids[i, j + 1], c='black', s=25, marker="x", zorder=2)


def plot_GMM_dist_data(x, y, df, n_shots, ball_col, receiver_position='receiver_start', only_half=True, compressed=True,
                       big_buckets=False, title='GMM Data'):
    chosen_shots = pd.DataFrame()
    ax = plt.gca()

    for i in range(n_shots):
        feasible_shots = df[
            (df.x02 > x[i] - 0.25) & (df.x02 < x[i] + 0.25) & (df.y02 > y[i] - 0.25) & (df.y02 < y[i] + 0.25)]
        if feasible_shots.shape[0] != 0:
            chosen_shots = chosen_shots.append(feasible_shots.sample())

    plot1D(data=chosen_shots, alpha_ball=1.0, alpha_player=0, ball_col=ball_col, only_half=True, compressed=True,
           receiver_position=receiver_position, title=title)
    createCourt(plt.gca(), only_half, compressed, big_buckets)

    return chosen_shots['vast'].mean()


def plot1D(data, centroids=None, means_tmp=None, just_ball=False, alpha_player=1, alpha_ball=1, ball_col='blue',
           compressed=False, only_half=False, big_buckets=False, receiver_position='receiver_start', title=""):
    # fig = plt.figure()
    # ax = plt.subplot(111)
    if not centroids is None:
        if just_ball == True:
            plt.scatter(data.x02, data.y02, c=colour_theme[data.Cluster], zorder=2, alpha=alpha_ball)
        else:
            plt.scatter(data.x02, data.y02, c=colour_theme[data.Cluster], zorder=2, alpha=alpha_ball)
            plt.scatter(data.px0, data.py0, c=colour_theme[data.Cluster], zorder=2, alpha=alpha_player)
            plt.scatter(data.ox0, data.oy0, c=colour_theme[data.Cluster], zorder=2, alpha=alpha_player)
        plotCentroids(centroids, means_tmp, just_ball)
    else:
        plt.scatter(data.x02, data.y02, c=ball_col, zorder=2, alpha=alpha_ball)
        if only_half == False:
            plt.scatter(data.px0, data.py0, c='red', zorder=2, alpha=alpha_player)
        if receiver_position == 'receiver_start':
            plt.scatter(data.ox0, data.oy0, c='purple', zorder=2, alpha=alpha_player)
        elif receiver_position == 'receiver_end':
            data = data.rename(columns={'opponent.end.x': 'opponent_end_x', 'opponent.end.y': 'opponent_end_y'})
            plt.scatter(data.opponent_end_x, data.opponent_end_y, c='orange', zorder=2, alpha=alpha_player)

    plt.title(title)
    createCourt(plt.gca(), compressed, only_half, big_buckets)


def plot2D(data, ax=plt.gca(), x_rot=0, col_ball='Blue', col_p1='Red', col_p2='Purple', title="", compressed=False,
           only_half=False, big_buckets=False):
    # fig = plt.figure()
    # grid = plt.GridSpec(2, 2)
    # ax = plt.subplot(grid[0, 0])
    for i in data.index:
        time1 = np.linspace(0, data.loc[i, 't1'], 1000)
        xline1 = data.loc[i, 'x3'] * (time1 ** 3) + data.loc[i, 'x2'] * (time1 ** 2) + data.loc[i, 'x1'] * (time1) + \
                 data.loc[i, 'x0']
        yline1 = data.loc[i, 'y3'] * (time1 ** 3) + data.loc[i, 'y2'] * (time1 ** 2) + data.loc[i, 'y1'] * (time1) + \
                 data.loc[i, 'y0']

        time2 = np.linspace(0, data.loc[i, 't2'], 1000)
        xline2 = data.loc[i, 'x32'] * (time2 ** 3) + data.loc[i, 'x22'] * (time2 ** 2) + data.loc[i, 'x12'] * (time2) + \
                 data.loc[i, 'x02']
        yline2 = data.loc[i, 'y32'] * (time2 ** 3) + data.loc[i, 'y22'] * (time2 ** 2) + data.loc[i, 'y12'] * (time2) + \
                 data.loc[i, 'y02']

        xfull = np.concatenate((xline1, xline2))
        yfull = np.concatenate((yline1, yline2))

        time3 = np.linspace(0, data.loc[i, 'duration'], 1000)
        xImpactPlayer = data.loc[i, 'px1'] * time3 + data.loc[i, 'px0']
        yImpactPlayer = data.loc[i, 'py1'] * time3 + data.loc[i, 'py0']
        xReceiverPlayer = data.loc[i, 'ox1'] * time3 + data.loc[i, 'ox0']
        yReceiverPlayer = data.loc[i, 'oy1'] * time3 + data.loc[i, 'oy0']

        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(x_rot)

        plt.plot(xfull, yfull, c=col_ball, transform=rot + base)
        plt.scatter(data.loc[i, 'x02'], data.loc[i, 'y02'], c=col_ball, s=15, transform=rot + base, zorder=2)
        plt.plot(xImpactPlayer, yImpactPlayer, col_p1, transform=rot + base)
        plt.scatter(data.loc[i, 'px0'], data.loc[i, 'py0'], c=col_p1, s=15, transform=rot + base, zorder=2)
        plt.plot(xReceiverPlayer, yReceiverPlayer, col_p2, transform=rot + base)
        plt.scatter(data.loc[i, 'ox0'], data.loc[i, 'oy0'], c=col_p2, s=15, transform=rot + base, zorder=2)

    plt.title(title)
    createCourt(ax, compressed, only_half, big_buckets)


def plot3D(data, ax):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    for i in data.index:
        time1 = np.linspace(0, data.loc[i, 't1'], 1000)
        xline1 = data.loc[i, 'x3'] * (time1 ** 3) + data.loc[i, 'x2'] * (time1 ** 2) + data.loc[i, 'x1'] * (time1) + \
                 data.loc[i, 'x0']
        yline1 = data.loc[i, 'y3'] * (time1 ** 3) + data.loc[i, 'y2'] * (time1 ** 2) + data.loc[i, 'y1'] * (time1) + \
                 data.loc[i, 'y0']
        zline1 = data.loc[i, 'z3'] * (time1 ** 3) + data.loc[i, 'z2'] * (time1 ** 2) + data.loc[i, 'z1'] * (time1) + \
                 data.loc[i, 'z0']

        time2 = np.linspace(0, data.loc[i, 't2'], 1000)
        xline2 = data.loc[i, 'x32'] * (time2 ** 3) + data.loc[i, 'x22'] * (time2 ** 2) + data.loc[i, 'x12'] * (time2) + \
                 data.loc[i, 'x02']
        yline2 = data.loc[i, 'y32'] * (time2 ** 3) + data.loc[i, 'y22'] * (time2 ** 2) + data.loc[i, 'y12'] * (time2) + \
                 data.loc[i, 'y02']
        zline2 = data.loc[i, 'z32'] * (time2 ** 3) + data.loc[i, 'z22'] * (time2 ** 2) + data.loc[i, 'z12'] * (time2) + \
                 data.loc[i, 'z02']

        xfull = np.concatenate((xline1, xline2))
        yfull = np.concatenate((yline1, yline2))
        zfull = np.concatenate((zline1, zline2))

        time3 = np.linspace(0, data.loc[i, 'duration'], 1000)
        xImpactPlayer = data.loc[i, 'px1'] * time3 + data.loc[i, 'px0']
        yImpactPlayer = data.loc[i, 'py1'] * time3 + data.loc[i, 'py0']
        xReceiverPlayer = data.loc[i, 'ox1'] * time3 + data.loc[i, 'ox0']
        yReceiverPlayer = data.loc[i, 'oy1'] * time3 + data.loc[i, 'oy0']

        ax.plot3D(xfull, yfull, zfull, 'blue')
        ax.plot3D(xImpactPlayer, yImpactPlayer, 'red')
        ax.plot3D(xReceiverPlayer, yReceiverPlayer, 'purple')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


def oldFunctions():
    # Old way of creating heatmap matrix
    # xLandBin = (pd.cut(ShotDataVast.x02, xBin, labels=False, retbins=True, right=False))[0]
    # yLandBin = (pd.cut(ShotDataVast.y02, yBin, labels=False, retbins=True, right=False))[0]
    #
    # ShotDataVast['xLandBin'] = (xBin[xLandBin].transpose())
    # ShotDataVast['yLandBin'] = (yBin[yLandBin].transpose())
    # # ShotDataVast['LandBucket'] = (yLandBin*len(xBin) + xLandBin)
    #
    # vast_hm = ShotDataVast.pivot_table(index = 'yLandBin', columns= 'xLandBin', values = 'vast', aggfunc='count')
    #
    # temp = pd.DataFrame(np.nan,index = np.flip(yBin), columns= xBin )
    # for i in vast_hm.axes[0]:
    #     for j in vast_hm.axes[1]:
    #         temp.loc[i,j] = (vast_hm.loc[i,j])
    lol = 5
