# coding=utf-8
import matplotlib.pyplot as plt
import os, ast
import numpy as np
from scipy.signal import savgol_filter
from itertools import groupby

def get_ongoing_metrics(file, path):
    # indices calculated and saved
    text = open(path + '/' + file + '.txt', "r")
    text = text.read()

    text_splt = text.split('\n')
    lst_tmp, flst_tmp = [], []
    for line in text_splt:
        try:
            folder_tmp = line.split(' ')[3]
            value_tmp = [float(s) for s in line.split(' ')[4:] if s[-1].isdigit()]
            if not value_tmp:
                # if list empty
                pass
            else:
                lst_tmp.append(value_tmp)
                flst_tmp.append(folder_tmp)
        except:
            pass
    # get data matrix
    value_arr = np.array(lst_tmp)
    folder_arr = np.array(flst_tmp)

    # get variable titles
    data = {}
    titles_tmp = [s for s in text_splt[1].split(' ')[4:] if not s[0].isdigit()]
    titles_tmp = ''.join(titles_tmp).split('---')
    titles_tmp = ''.join(titles_tmp).split(':')
    titles_tmp = titles_tmp[:value_arr.shape[1]]

    # collect variables and related values
    for i in range(value_arr.shape[1]):
        data[titles_tmp[i]] = value_arr[:, i].tolist()  # data must be non-negative

    return data, folder_arr


def compare_metrics(data, vars, colors, lw, xsteps, ysteps, xlim, ylim, training, validation, grid, path,
                    train_norm=(), val_norm=(), save=False):
    fig = plt.figure()

    cnt = 0
    # iterate through different training curves
    training_dict, validation_dict = {}, {}
    for d in data:
        x1, _ = get_ongoing_metrics('training', d)
        if validation is True:
            x2, _ = get_ongoing_metrics('validation', d)

        sub_cnt = 0
        # iterate through variables of training curve
        for var in vars:
            try:
                # skip variable if it doesn't exist for the model analyzed
                _ = x1[var]
            except:
                sub_cnt += 1
                continue

            xlim_tmp = xlim[sub_cnt]
            ylim_tmp = ylim[sub_cnt]

            __ = fig.add_subplot(1, len(vars), sub_cnt + 1)
            if training is True:
                x1_tmp = x1[var][:xlim_tmp]
                if (not train_norm) is False:
                    # if vector tuple norm is not empty, then filter validation data
                    # Note: polynomial used to fit the samples. polyorder must be less than window_length.
                    x1_tmp = savgol_filter(x1_tmp, window_length=train_norm[0], polyorder=train_norm[1])
                __ = plt.plot(range(len(x1_tmp)), x1_tmp, linewidth=lw[sub_cnt], color=colors[cnt])
            if validation is True:
                x2_tmp = x2[var][:xlim_tmp]
                if (not val_norm) is False:
                    # if vector tuple norm is not empty, then filter validation data
                    # Note: polynomial used to fit the samples. polyorder must be less than window_length.
                    x2_tmp = savgol_filter(x2_tmp, window_length=val_norm[0], polyorder=val_norm[1])
                __ = plt.plot(range(len(x2_tmp)), x2_tmp,
                              linewidth=0.7, linestyle='--', color=colors[cnt], alpha=0.5)

            __, __ = plt.xticks(np.arange(0, xlim_tmp, step=xsteps[sub_cnt]))
            __, __ = plt.yticks(np.arange(0, ylim_tmp, step=ysteps[sub_cnt]))
            plt.ylim(top=ylim_tmp)

            plt.title(var)
            plt.grid(grid)

            # store data
            training_dict[var + '_' + str(cnt + 1)] = x1[var][:xlim_tmp]
            if validation is True:
                validation_dict[var + '_' + str(cnt + 1)] = x2[var][:xlim_tmp]

            sub_cnt += 1

        cnt += 1

    if save is True:
        img_name = "metrics_vars-{}_types-{}_training-{}" \
                   "_validation-{}".format(len(data), '-'.join(vars), training, validation)
        output_name = os.path.join(os.path.abspath(path), img_name)
        plt.savefig(output_name, dpi=300)

    plt.show()
    if validation is True:
        return training_dict, validation_dict
    else:
        return training_dict


def boxplot_by_folder(root_path, train_folders, metric, scatter, line_value, data_limit=99999):
    group_list, folder_list = [], []
    fig = plt.figure()
    for fld in train_folders:
        data, folders = get_ongoing_metrics('training', root_path+fld)
        if scatter:
            plt.scatter(folders, data[metric], s=1, c='black')
        else:
            X = np.array([folders, data[metric]]).T
            Y = X[X[:, 0].argsort()]
            groups = []
            for key, group in groupby(Y, lambda x: x[0]):
                groups.append(np.array([list(g) for g in group]))
            for grp in groups:
                group_list.append(list(grp[:, 1][:data_limit].astype('float32')))
                folder_list.append(grp[0,0])
    if not scatter:
        plt.boxplot(group_list, labels=folder_list, vert=True)
        left, right = plt.xlim()
        plt.hlines(line_value, xmin=left, xmax=right, colors='black', linestyles=':')
    plt.show()


boxplot_by_folder('./checkpoint/', ['train_4/','train_5/','train_6/','train_7/','train_8/','train_9/'], 'iou',
                  scatter=True, line_value=0.5, data_limit=9999999)

# Compare loss metrics
titles = ['loss', 'loss_sem', 'loss_center', 'sem_acc']
c1 = './checkpoint/train_1/'
mult = 10
trainingD, validationD = compare_metrics(data=[c1], vars=titles,
                                         colors=['#2f5c6f', '#f96552', '#6cb6bf', '#a1a1a1', '#000000'],
                                         lw=[1.] * mult, xsteps=[100] * mult, ysteps=[0.1] * mult, ylim=[1, 1, 1, 1],
                                         xlim=[30] * mult,
                                         training=True, validation=True, grid=True,
                                         train_norm=(), val_norm=(),
                                         path='./train/', save=False)

# Find best epoch given validation loss
d = 1
intv = [120, 230]
diff = np.array(validationD['loss_cls_' + str(d)][intv[0]:intv[1]])
idx = intv[0] + np.argmin(diff)
print("Step: ", idx)
print("Best training loss: ", np.array(trainingD['loss_cls_' + str(d)][idx]))
print("Best validation loss: ", np.array(validationD['loss_cls_' + str(d)][idx]))
print("Best training f1: ", np.array(trainingD['f0_' + str(d)][idx]))
print("Best validation f1: ", np.array(validationD['f0_' + str(d)][idx]))
print("Training Rsp: ", np.array(trainingD['ratio_conv_' + str(d)][idx]))
