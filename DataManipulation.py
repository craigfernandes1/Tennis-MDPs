import numpy as np
import pandas as pd
import math
from PlottingFunction import createBins



def filter_data_by_buckets(df, player, buckets, compressed=False, only_half=False, big_buckets=False):
    xBin, yBin = createBins(compressed, only_half, big_buckets)
    df_return = pd.DataFrame()
    df = df.rename(columns={'opponent.end.x': 'opponent_end_x', 'opponent.end.y': 'opponent_end_y'})

    for bucket in buckets:
        # x = (bucket % (len(xBin) - 1)) - 1
        # y = math.floor(bucket / (len(xBin) - 1))

        # labelling court vertically
        y = (bucket % (len(yBin) - 1)) - 1
        x = math.floor(bucket / (len(yBin) - 1))

        xleft = xBin[x]
        xright = xBin[x + 1]
        ybottom = yBin[y]
        ytop = yBin[y + 1]

        if player == "impact":
            df_return = df_return.append(
                df[(df.px0 > xleft) & (df.px0 < xright) & (df.py0 > ybottom) & (df.py0 < ytop)])
        elif player == "receiver_start":
            df_return = df_return.append(
                df[(df.ox0 > xleft) & (df.ox0 < xright) & (df.oy0 > ybottom) & (df.oy0 < ytop)])
        elif player == 'receiver_end':
                df_return = df_return.append(
                    df[(df.opponent_end_x > xleft) & (df.opponent_end_x < xright) & (df.opponent_end_y > ybottom) & (df.opponent_end_y < ytop)])

    return df_return

def keep_data_in_frame(df, compressed=False, only_half=False, big_buckets=False):
    if big_buckets == True:
        df = df[(df.x02 < 15.8872) & (df.x02 > -15.8872) & (df.y02 < 7) & (df.y02 > -7)  # keep ball landing in frame
                & (df.px0 > -15.8872) & (df.py0 < 7) & (df.py0 > -7)  # keep impact player in frame
                & (df.ox0 < 15.8872) & (df.oy0 < 7) & (df.oy0 > -7)  # keep receiving player in frame
                & (df.px0 < 0) & (df.ox0 > 0)]  # keep both players on their side of the court
    if compressed == True and big_buckets == False:
        df = df[(df.x02 < 13.8872) & (df.x02 > -13.8872) & (df.y02 < 7.4864) & (df.y02 > -7.4864)  # keep ball landing in frame
                & (df.px0 > -13.8872) & (df.py0 < 7.4864) & (df.py0 > -7.4864)  # keep impact player in frame
                & (df.ox0 < 13.8872) & (df.oy0 < 7.4864) & (df.oy0 > -7.4864)  # keep receiving player in frame
                & (df.px0 < 0) & (df.ox0 > 0)]  # keep both players on their side of the court
    if compressed == False and big_buckets== False:
        df = df[(df.x02 < 18.8872) & (df.x02 > -18.8872) & (df.y02 < 9.4864) & (df.y02 > -9.4864)  # keep ball landing in frame
                & (df.px0 > -18.8872) & (df.py0 < 9.4864) & (df.py0 > -9.4864)  # keep impact player in frame
                & (df.ox0 < 18.8872) & (df.oy0 < 9.4864) & (df.oy0 > -9.4864)  # keep receiving player in frame
                & (df.px0 < 0) & (df.ox0 > 0)]  # keep both players on their side of the court
    if only_half == True:
        df = df[df.x02 > 0]

    return df

def remove_error_shots(df,type):
    if type == 'ad_serve':
        return df[(df.x02 < 6.4008) & (df.y02 > 0) & (df.y02 < 823 / 200) & (df.error == 0)]
    elif type == 'deuce_serve':
        return df[(df.x02 < 6.4008) & (df.y02 < 0) & (df.y02 > -823 / 200) & (df.error == 0)]
    elif type == 'rally':
        return df[(df.x02 < 11.8872) & (df.y02 > -823 / 200) & (df.y02 < 823 / 200) & (df.error == 0)]

def change_vast_error_shots(df,type):
    if type == 'ad_serve':
        df['vast'] = np.where(((df.x02 > 6.4008) | (df.y02 > 0) | (df.y02 < -823 / 200) | (df.error == 1)), 0,
                              df.vast)
    elif type == 'deuce_serve':
        df['vast'] = np.where(((df.x02 > 6.4008) | (df.y02 < 0) | (df.y02 > 823 / 200) | (df.error == 1)), 0,
                              df.vast)
    elif type == 'rally':
        df['vast'] = np.where(((df.x02 > 11.8872) | (df.y02 < -823 / 200) | (df.y02 > 823 / 200) | (df.error == 1)), 0,
                              df.vast)
    return df

def bucket_centroid(bucket, compressed=False, only_half=False, big_buckets=False):
    xBin, yBin = createBins(compressed, only_half, big_buckets)

    # x = (bucket % (len(xBin) - 1)) - 1
    # y = math.floor(bucket / (len(xBin) - 1))

    # labelling court vertically
    y = (bucket % (len(yBin) - 1)) - 1
    x = math.floor(bucket / (len(yBin) - 1))

    xleft = xBin[x]
    xright = xBin[x + 1]
    ybottom = yBin[y]
    ytop = yBin[y + 1]

    xmiddle = (xright + xleft) / 2
    ymiddle = (ytop + ybottom) / 2

    return (xmiddle,ymiddle)

def get_peak_vast_heatmap(heatmap):
    heatmap = pd.DataFrame(heatmap)
    peak = heatmap.max().max()
    peak_idx = heatmap.where(heatmap == peak).dropna(how='all').dropna(axis=1)
    row = peak_idx.index.values
    col = peak_idx.columns.values
    xBin, yBin = createBins(compressed=True, only_half=True)
    return (((xBin[row[0]] + xBin[row[0] + 1]) / 2), ((yBin[col[0]] + yBin[col[0] + 1]) / 2))

def get_starting_bucket(df, player):
    xBin, yBin = createBins(compressed=True, only_half=False, big_buckets=True)

    if player == 'impact':
        xBinStart = (pd.cut(df.px0, xBin, labels=False, retbins=True, right=False))[0]
        yBinStart = (pd.cut(df.py0, yBin, labels=False, retbins=True, right=False))[0]
    else:
        xBinStart = (pd.cut(df.ox0, xBin, labels=False, retbins=True, right=False))[0]
        yBinStart = (pd.cut(df.oy0, yBin, labels=False, retbins=True, right=False))[0]

    return (xBinStart * (len(yBin) - 1) + yBinStart) + 1

def get_ending_bucket(df, player):
    xBin, yBin = createBins(compressed=True, only_half=False, big_buckets=True)
    df = df.rename(columns={'player.end.x': 'player_end_x', 'player.end.y': 'player_end_y'})
    df = df.rename(columns={'opponent.end.x': 'opponent_end_x', 'opponent.end.y': 'opponent_end_y'})

    if player == 'impact':
        xBinStart = (pd.cut(df.player_end_x, xBin, labels=False, retbins=True, right=False))[0]
        yBinStart = (pd.cut(df.player_end_y, yBin, labels=False, retbins=True, right=False))[0]
    else:
        xBinStart = (pd.cut(df.opponent_end_x, xBin, labels=False, retbins=True, right=False))[0]
        yBinStart = (pd.cut(df.opponent_end_y, yBin, labels=False, retbins=True, right=False))[0]

    return (xBinStart * (len(yBin) - 1) + yBinStart) + 1

def reformatVector(vect):
    serve = vect[0:1764, 0]
    servereturn = vect[1764:3528, 0]
    rally = vect[3528:5292, 0]

    serve = np.reshape(serve, (42, 42))
    servereturn = np.reshape(servereturn, (42, 42))
    rally = np.reshape(rally, (42, 42))

    return {'serve': serve, 'servereturn': servereturn, 'rally': rally}


def add_landing_region_to_df(df,more_actions=False):
    df_rally, df_serve = [x for _, x in df.groupby(df['Type'] == 'serve')]
    df_serve = df_serve.sort_values(by=['adpoint'])
    df_serve = df_serve.reset_index(drop=True)
    df_deuceserve, df_adserve = [x for _, x in df_serve.groupby(df_serve['adpoint'] == 1)]

    df_adserve['region'] = landing_region_by_type(df_adserve, more_actions, 'ad_serve')
    df_deuceserve['region'] = landing_region_by_type(df_deuceserve, more_actions, 'deuce_serve')
    df_rally['region'] = landing_region_by_type(df_rally, more_actions, 'rally')

    return pd.concat([df_deuceserve, df_adserve, df_rally], ignore_index=True)

def landing_region_by_type(df, more_actions, type):


    if type == 'ad_serve':
        xBin = [0, 16]
        yBin = [-7.5, -823 / 400, 7.5]
        region_dict = {0: 'Error', 1: 'serve_ad_corner', 2: 'serve_ad_middle'}
    elif type == 'deuce_serve':
        xBin = [0, 16]
        yBin = [-7.5, 823 / 400, 7.5]
        region_dict = {0: 'Error', 1: 'serve_deuce_middle', 2: 'serve_deuce_corner'}
    elif type == 'rally':
        if more_actions == False:
            xBin = [-2.1336, 6.4, 16]
            yBin = [-7.5, -823 / 400, 823 / 400, 7.5]

            region_dict = {0: 'Error', 1: 'rally_short_ad', 2: 'rally_short_middle', 3: 'rally_short_deuce',
                           4: 'rally_deep_ad', 5: 'rally_deep_middle', 6: 'rally_deep_deuce'}
        else:
            xBin = [-16, 0, 2.1336, 4.2672, 6.4, 8.5328, 10.8872, 11.8872, 16]
            yBin = [-7.5, -823 / 200, -623 / 200, 0, 623 / 200, 823 / 200, 7.5]

            df_new = df.assign(
                xCut=(pd.cut(df.x02, xBin, labels=False, retbins=True, right=False))[0],
                yCut=(pd.cut(df.y02, yBin, labels=False, retbins=True, right=False))[0])

            df_new['regionCord'] = df_new['xCut'].astype(str) + '_' + df_new['yCut'].astype(str)

            region_dict = {'0_0': 'OBDrop', '0_1': 'OBDrop', '0_2': 'OBDrop', '0_3': 'OBDrop', '0_4': 'OBDrop',
                           '0_5': 'OBDrop',
                           '1_0': 'adOBShortSide', '1_1': 'adDrop', '1_2': 'adDrop', '1_3': 'deuceDrop',
                           '1_4': 'deuceDrop', '1_5': 'deuceOBShortSide',
                           '2_0': 'adOBShortSide', '2_1': 'adShortSide', '2_2': 'adShort', '2_3': 'deuceShort',
                           '2_4': 'deuceShortSide', '2_5': 'deuceOBShortSide',
                           '3_0': 'adOBShortSide', '3_1': 'adShortSide', '3_2': 'adShortMid', '3_3': 'deuceShortMid',
                           '3_4': 'deuceShortSide', '3_5': 'deuceOBShortSide',
                           '4_0': 'adOBDeepSide', '4_1': 'adDeepSide', '4_2': 'adMidDeep', '4_3': 'deuceMidDeep',
                           '4_4': 'deuceDeepSide', '4_5': 'deuceOBDeepSide',
                           '5_0': 'adOBDeepSide', '5_1': 'adDeepSide', '5_2': 'adDeep', '5_3': 'deuceDeep',
                           '5_4': 'deuceDeepSide', '5_5': 'deuceOBDeepSide',
                           '6_0': 'adOBDeepSide', '6_1': 'adCorner', '6_2': 'adBase', '6_3': 'deuceBase',
                           '6_4': 'deuceCorner', '6_5': 'deuceOBDeepSide',
                           '7_0': 'adOBBase', '7_1': 'adOBBase', '7_2': 'adOBBase', '7_3': 'deuceOBBase',
                           '7_4': 'deuceOBBase', '7_5': 'deuceOBBase'}

            df_new["regionCord"].replace(region_dict, inplace=True)

            return df_new['regionCord']

    xBinStart = (pd.cut(df.x02, xBin, labels=False, retbins=True, right=False))[0]
    yBinStart = (pd.cut(df.y02, yBin, labels=False, retbins=True, right=False))[0]

    region = (xBinStart * (len(yBin) - 1) + yBinStart) + 1
    region = np.nan_to_num(region)
    region = [region_dict[shot] for shot in region]

    return region


def actionProbabilityVector(df, distribution_region_dict, region_labels, weights_rally, weights_deuce_serve, weights_ad_serve):
    df_rally, df_serve = [x for _, x in df.groupby(df['Type'] == 'serve')]
    df_serve = df_serve.sort_values(by=['adpoint'])
    df_serve = df_serve.reset_index(drop=True)
    df_deuceserve, df_adserve = [x for _, x in df_serve.groupby(df_serve['adpoint'] == 1)]

    df_deuceserve = specificActionProbabilityVector(df_deuceserve, distribution_region_dict, region_labels[27:29],weights_deuce_serve)
    df_adserve = specificActionProbabilityVector(df_adserve, distribution_region_dict, region_labels[25:27], weights_ad_serve)
    df_rally = specificActionProbabilityVector(df_rally, distribution_region_dict, region_labels[0:25], weights_rally)

    df_full = pd.concat([df_deuceserve, df_adserve, df_rally], axis=0)
    df_full.replace(np.nan, 0, inplace=True)

    return df_full


def specificActionProbabilityVector(df, distribution_dict, region_labels, weights):
    prob = np.asarray([distribution_dict[region].pdf(df[['x02', 'y02']]) for region in region_labels])
    prob = pd.DataFrame(np.transpose(prob), columns=region_labels, index=df.index)
    sum = (prob * weights).sum(axis=1)
    prob = (prob * weights).divide(sum, axis=0)

    return pd.concat([df, prob], axis=1)