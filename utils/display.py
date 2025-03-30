import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib import rcParams


# plt.rcParams['figure.figsize'] = (6.8, 2.6)
plt.rcParams['figure.figsize'] = (8.4, 6.8)
# 全局设置字体及大小，设置公式字体即可，若要修改刻度字体，可在此修改全局字体
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['Times New Roman + SimSun'],
    "font.size": 10,  # 字号，大家自行调节
    'axes.unicode_minus': False  # 处理负号，即-号
}
plt.rcParams.update(config)


def display_heatmap(values: list[list], labels: list = None, map_title: str = 'Heatmap', save_path: str = None) -> None:
    if (ticks := labels) is None:
        ticks = 'auto'
    ax = sns.heatmap(values, cmap="YlGnBu", xticklabels='auto', yticklabels=ticks, annot=True,
                     linewidths=.5)  # 修改颜色，添加线宽
    # ax.set_xticklabels(labels=ticks, rotation=60, ha='right', rotation_mode='anchor')  # 放大横轴坐标并逆时针旋转45°
    ax.set_title(map_title)  # 图标题
    ax.set_xlabel('x label')  # x轴标题
    ax.set_ylabel('y label')
    plt.tight_layout()
    plt.show()
    if save_path:
        figure = ax.get_figure()
        figure.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)


def display_scatter(x: list[list], labels: list = None, map_title: str = 'Scattermap', save_path: str = None) -> None:
    x = np.array(x)
    assert x.shape[-1] == 2, "The param [x]'s second dim is not 2."

    fig, ax = plt.subplots()
    if labels is None:
        plt.scatter(x[:, 0], x[:, 1])
    else:
        assert len(labels) == x.shape[0], "There are some discord between param [x] and [labels]."
        labels = np.array(labels)
        mask_list = ['o', '^', 's', 'D', '*', '+']
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            idx = labels == label
            plt.scatter(x[idx, 0], x[idx, 1], label=f'{label}', alpha=0.6, marker=mask_list[int(i // 10)])

    ax.set_title(map_title)
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    ax.legend()
    plt.tight_layout()
    plt.show()
    if save_path:
        figure = ax.get_figure()
        figure.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)


def display_range_fluctuation_curve(values_dict: dict, labels: list = None, decoration_dict: dict = None,
                                    map_title: str = 'Range Fluctuation Curve', save_path: str = None,
                                    xlabel: str = None, ylabel: str = None) -> None:
    """

    :param values_dict:
                        {
                            'logical network':
                                [[0.95, 0.98, 0.98, 0.92, 0.25],
                                 [0.98, 0.90, 0.75, 0.88, 0.35],
                                 [0.91, 0.85, 0.88, 0.51, 0.10],],
                            'SWRL':
                                [[0.85, 0.88, 0.88, 0.82, 0.85],
                                 [0.88, 0.80, 0.65, 0.88, 0.65],
                                 [0.81, 0.75, 0.78, 0.61, 0.80],]
                        }
    :param labels:      ['Backcover', 'PCBcover', 'PCB', 'Carrier', 'LCDmodule']
    :param decoration_dict:
                        {
                            'logical network': {'marker': 's', 'c': 'r', 'linestyle': '--'},
                            'SWRL': {'marker': '^', 'c': 'g'}
                        }
    :param map_title:
    :param save_path:
    :return:
    """
    fig, ax = plt.subplots()
    decoration_list = ['marker', 'c', 'linestyle']

    for name, values_list in values_dict.items():
        x = np.array(values_list)
        x_mean = np.mean(x, axis=0)
        x_max = np.max(x, axis=0)
        x_min = np.min(x, axis=0)
        x_num = x_mean.shape[0]

        gt_decoration = {d: None for d in decoration_list}
        if decoration_dict and name in decoration_dict.keys():
            decoration = decoration_dict[name]
            gt_decoration = {d: decoration[d] if d in decoration.keys() else None for d in decoration_list}

        plt.plot(list(range(x_num)), x_mean, label=name, linewidth=2,
                 marker=gt_decoration['marker'], c=gt_decoration['c'], linestyle=gt_decoration['linestyle'])
        plt.fill_between(
            list(range(x_num)),  # x
            x_max,  # y upper
            x_min,  # y lower
            color=gt_decoration['c'],  # color = edgecolor + facecolor
            edgecolor='none',  # 不要边界
            linewidth=2,  # 浅色区边界粗细
            alpha=0.2  # 浅色区能见度
        )

    ax.set_title(map_title)
    if not labels:
        labels = list(range(x_num))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(handlelength=3, fancybox=True, framealpha=0)  # 透明图例
    plt.tight_layout()
    plt.show()
    if save_path:
        figure = ax.get_figure()
        figure.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)  # 透明背景


def display_multi_bar(values_dict: dict, labels: list = None, type: str = 'var',
                      map_title: str = 'Multi Bar Chart', save_path: str = None,
                      xlabel: str = None, ylabel: str = None) -> None:
    """

    :param type: choose errbar type
    :param values_dict:
                    {
                        'Backcover': [
                            [0.90, 0.80, 0.10, 0.45],[0.91, 0.81, 0.11, 0.41],[0.92, 0.82, 0.12, 0.42],
                        ],
                        'PCBcover': [
                            [0.80, 0.70, 0.50, 0.35],[0.82, 0.72, 0.52, 0.32],[0.83, 0.73, 0.53, 0.33],
                        ],
                        'PCB': [
                            [0.70, 0.80, 0.40, 0.45],[0.71, 0.81, 0.41, 0.41],[0.75, 0.85, 0.45, 0.45],
                        ],
                        'Carrier': [
                            [0.60, 0.10, 0.50, 0.05],[0.65, 0.15, 0.55, 0.05],[0.68, 0.18, 0.58, 0.08],
                        ],
                        'LCDmodule': [
                            [0.50, 0.30, 0.60, 0.15],[0.59, 0.30, 0.60, 0.19],[0.56, 0.30, 0.12, 0.16],
                        ],
                    }
    :param labels:  ['Color', 'Geometry', 'State', 'Size']
    :param map_title:
    :param save_path:
    :return:
    """
    init_values = np.array([v for v in values_dict.values()])
    values_mean = np.mean(init_values, axis=1)
    if type == 'diff':
        values_max = np.max(init_values, axis=1)
        values_min = np.min(init_values, axis=1)
        low_err = (values_mean - values_min).T
        high_err = (values_max - values_mean).T
    elif type == 'var':
        values_var = np.var(init_values, axis=1)
        low_err = high_err = values_var.T
        values_max = values_var + values_mean
    elif type == 'std':
        values_std = np.std(init_values, axis=1)
        low_err = high_err = values_std.T
        values_max = values_std + values_mean
    elif type == 'weight':
        # last column in last dim is weight!
        weights = init_values[:, :, -1]
        init_values = init_values[:, :, :-1]
        reweight_values = init_values * weights[:, :, None]
        values_max = values_mean = np.clip(np.sum(reweight_values, axis=1), None, 1.0)
    else:
        raise ValueError(f"wrong type with {type}.")
    if type != 'weight':
        err = np.array([[list(low_err[i]), list(high_err[i])] for i in range(low_err.shape[0])])
    # print(err)

    values_max = values_max.T
    values = values_mean.T
    name = [k for k in values_dict.keys()]
    # print(values)
    assert values.shape[0] == len(labels), "There are some discord between [values_dict] and [labels]."

    fig, ax = plt.subplots()
    width = 0.5 #0.3  0.5  0.1
    x = np.array([(width * (len(labels) + 1)) * i for i in range(len(values_dict.keys()))])
    # x = np.arange(len(values_dict.keys()))
    # len(values_dict.keys())
    xs = np.array([x + width * i for i in range(init_values.shape[-1])])
    x_mean = np.mean(xs, axis=0) - width / 2 + width / 2
    # print(x_mean)

    # colors = ['#D8EBCD', '#B5D99F', '#89C266', '#60993D', '#3A5D25']
    colors = ['#FFE18B', '#FFCF47']
    # colors = colors[-2:]
    # print(colors)
    assert values.shape[0] <= len(colors), "Please add color in [colors]."

    for i in range(values.shape[0]):
        plt.bar(xs[i], values[i], width=width*3/4, label=labels[i], color=colors[i], alpha=0.8)
        if type != 'weight':
            plt.errorbar(xs[i], values[i], yerr=err[i], fmt='none', color='black', capsize=5)
        for j in range(values.shape[-1]):
            plt.text(xs[i][j], values_max[i][j], round(values[i][j], 2), va="bottom", ha="center")

    # for i in range(len(x)):
    #     if i == 0:
    #         continue
    #     plt.plot([x[i] - width, x[i] - width], [0, 1], linestyle='--', color='gray')

    ax.set_title(map_title)
    ax.set_xticks(x_mean)
    ax.set_xticklabels(name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fancybox=True, framealpha=0)  # 透明图例
    ax.legend(fancybox=True)
    plt.tight_layout()
    plt.show()
    if save_path:
        figure = ax.get_figure()
        figure.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)  # 透明背景

if __name__ == '__main__':
    values_dict = {
                            'logical network':
                                [[0.95, 0.98, 0.98, 0.92, 0.25],
                                 [0.98, 0.90, 0.75, 0.88, 0.35],
                                 [0.91, 0.85, 0.88, 0.51, 0.10],],
                            'SWRL':
                                [[0.85, 0.88, 0.88, 0.82, 0.85],
                                 [0.88, 0.80, 0.65, 0.88, 0.65],
                                 [0.81, 0.75, 0.78, 0.61, 0.80],]
                        }
    labels = ['Backcover', 'PCBcover', 'PCB', 'Carrier', 'LCDmodule']
    decoration_dict = {
                            'logical network': {'marker': 's', 'c': 'r', 'linestyle': '--'},
                            'SWRL': {'marker': '^', 'c': 'g'}
                        }
    display_range_fluctuation_curve(values_dict, decoration_dict=decoration_dict)
