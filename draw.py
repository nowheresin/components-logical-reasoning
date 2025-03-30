from matplotlib import pyplot as plt
from utils.display import display_range_fluctuation_curve, display_multi_bar

# Fig. 6: Distribution of attributes for misclassified components using the common logical network.
# multi_bar
values_dict = {
    'Front bezel': [[0.67, 0.67, 0.82, 0.69, 0.56, 0.16], [0.74, 0.58, 0.66, 0.62, 0.82, 0.2], [0.69, 0.59, 0.64, 0.66, 0.86, 0.22], [0.83, 0.58, 0.63, 0.73, 0.67, 0.21], [0.68, 0.59, 0.63, 0.65, 0.89, 0.21]],
    'Back cover': [[0.77, 0.8, 0.66, 0.56, 0.8, 0.19], [0.72, 0.71, 0.7, 0.68, 0.73, 0.27], [0.61, 0.67, 0.65, 0.94, 0.67, 0.19], [0.74, 0.74, 0.65, 0.74, 0.46, 0.14], [0.74, 0.69, 0.61, 0.8, 0.69, 0.21]],
    'Stand': [[0.72, 0.79, 0.79, 0.66, 0.88, 0.3], [0.81, 0.6, 0.73, 0.77, 0.67, 0.25], [0.8, 0.52, 0.78, 0.88, 0.45, 0.13], [0.71, 0.55, 0.88, 0.63, 0.39, 0.13], [0.84, 0.64, 0.71, 0.66, 0.62, 0.18]],
    'Base': [[0.58, 0.89, 0.85, 0.6, 0.52, 0.37], [0.84, 0.46, 0.44, 0.79, 0.82, 0.34], [0.9, 1.0, 1.0, 0.1, 0.0, 0.05], [1.0, 1.0, 1.0, 0.0, 0.0, 0.03], [0.56, 0.61, 0.78, 0.73, 0.51, 0.21]],
    'Circuit board': [[0.69, 0.72, 0.71, 0.66, 0.71, 0.22], [0.69, 0.72, 0.71, 0.66, 0.71, 0.22], [0.69, 0.72, 0.71, 0.66, 0.71, 0.22], [0.63, 1.0, 0.91, 0.59, 0.48, 0.12], [0.69, 0.72, 0.71, 0.66, 0.71, 0.22]],
    'PCB cover': [[0.56, 0.77, 0.44, 0.86, 0.9, 0.09], [0.75, 0.69, 0.68, 0.74, 0.66, 0.25], [0.75, 0.69, 0.68, 0.74, 0.66, 0.25], [0.81, 0.84, 0.64, 0.69, 0.49, 0.17], [0.75, 0.69, 0.68, 0.74, 0.66, 0.25]],
    'Carrier': [[0.65, 0.78, 0.85, 0.82, 0.73, 0.24], [0.77, 0.61, 0.77, 0.63, 0.68, 0.17], [0.78, 0.67, 0.82, 0.64, 0.72, 0.2], [0.78, 0.74, 0.63, 0.63, 0.66, 0.22], [0.87, 0.59, 0.74, 0.51, 0.7, 0.17]],
    'LCD module': [[0.95, 0.77, 0.74, 0.73, 0.73, 0.18], [0.73, 0.68, 0.7, 0.65, 0.69, 0.22], [0.73, 0.68, 0.7, 0.65, 0.69, 0.22], [0.75, 0.82, 0.6, 0.65, 0.58, 0.16], [0.73, 0.68, 0.7, 0.65, 0.69, 0.22]],
}
labels = ['Material', 'Color', 'Geometry', 'Size', 'State']
plt.rcParams['figure.figsize'] = (12.8, 6.3)  #(9.8, 4.8)
# display_multi_bar(values_dict, labels=labels, type='weight',
#                   map_title='Distribution of attributes for misclassified components using the common logical model.',
#                   xlabel='The component classes.', ylabel='Frequency.',
#                   )  #  save_path='./savefig/Fig_6.png'

# Fig. 7: Distribution of attributes for misclassified components using the sampling logical network.
# multi_bar
values_dict = {
    'Front bezel': [[1.0, 1.0, 1.0, 0.0, 0.0, 0.75], [0.99, 0.99, 0.99, 0.0, 0.0, 0.13], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.99, 0.99, 0.99, 0.0, 0.0, 0.13]],
    'Back cover': [[0.93, 0.79, 0.86, 0.75, 0.0, 0.33], [0.62, 1.0, 0.37, 1.0, 0.0, 0.09], [0.83, 0.78, 0.78, 0.61, 0.0, 0.27], [0.9, 1.0, 1.0, 0.1, 0.0, 0.12], [0.94, 1.0, 0.69, 0.44, 0.0, 0.19]],
    'Stand': [[0.79, 0.82, 0.93, 0.64, 0.61, 0.41], [0.73, 0.77, 0.88, 0.77, 0.5, 0.38], [1.0, 0.86, 0.71, 0.71, 0.0, 0.1], [1.0, 1.0, 0.5, 0.5, 0.0, 0.06], [1.0, 1.0, 0.0, 1.0, 0.0, 0.04]],
    'Base': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.5, 0.5, 0.0, 0.62], [1.0, 1.0, 0.0, 1.0, 0.0, 0.23], [1.0, 1.0, 1.0, 0.0, 0.0, 0.15], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    'Circuit board': [[0.17, 1.0, 1.0, 0.83, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    'PCB cover': [[1.0, 1.0, 1.0, 0.0, 0.0, 0.35], [1.0, 1.0, 1.0, 0.0, 0.0, 0.35], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    'Carrier': [[1.0, 1.0, 1.0, 0.0, 0.0, 0.1], [0.87, 0.78, 0.74, 0.96, 0.7, 0.74], [0.99, 0.99, 0.0, 0.99, 0.0, 0.03], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.75, 0.25, 1.0, 1.0, 0.0, 0.13]],
    'LCD module': [[1.0, 1.0, 1.0, 0.0, 0.0, 0.41], [1.0, 1.0, 1.0, 0.0, 0.0, 0.41], [1.0, 1.0, 1.0, 0.0, 0.0, 0.19], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
}
labels = ['Material', 'Color', 'Geometry', 'Size', 'State']
plt.rcParams['figure.figsize'] = (12.8, 6.3)  #(9.8, 4.8)
# display_multi_bar(values_dict, labels=labels, type='weight',
#                   map_title='Distribution of attributes for misclassified components using the sampling logical model.',
#                   xlabel='The component classes.', ylabel='Frequency.',
#                   )  #  save_path='./savefig/Fig_7.png'

# Fig. 8: Comparison of classification accuracy for the common logic network (common) and
# the sampling logic network (sampling) with and without domain-specific data processing.
# multi_bar
values_dict = {
    'Conventional': [[51.88, 50.70], [52.99, 43.96], [48.89, 52.05], [55.38, 62.50], [43.19, 49.03]],
    'Sampling': [[86.70, 96.81], [97.12, 96.98], [97.33, 98.61], [98.40, 99.20], [97.92, 99.17]],
}
labels = ['w/o data processing', 'w/ data processing']
plt.rcParams['figure.figsize'] = (5.4, 3.8)  #(9.8, 4.8)
# display_multi_bar(values_dict, labels=labels, type='std',
#                   map_title='The distribution.',
#                   xlabel='Type of model.', ylabel='ACC (%).',
#                   )  #  save_path='./savefig/Fig_8.png'

# Fig. 9: Comparison of classification accuracy for the common network (common) and
# the sampling network (sampling) with and without logical reasoning module.
# multi_bar
values_dict = {
    'Conventional': [[49.03, 50.70], [51.88, 43.96], [51.84, 52.05], [59.76, 62.50], [46.50, 49.03]],
    'Sampling': [[95.42, 96.81], [97.92, 96.98], [98.96, 98.61], [98.82, 99.20], [98.82, 99.17]],
}
labels = ['w/o logical reasoning module', 'w/ logical reasoning module']
plt.rcParams['figure.figsize'] = (5.4, 3.8)  #(9.8, 4.8)
# display_multi_bar(values_dict, labels=labels, type='std',
#                   map_title='The distribution.',
#                   xlabel='Type of model.', ylabel='ACC (%).',
#                   )  #  save_path='./savefig/Fig_9.png'

# Fig. 10: Performance of models with varying degrees of attribute sampling.
# range_fluctuation_curve
values_dict = {
    'models of different degree attribute sampling':[
        [50.70, 78.20, 96.81, 92.71],
        [43.96, 95.70, 96.98, 98.37],
        [52.05, 89.48, 98.61, 99.10],
        [62.50, 93.30, 99.20, 99.44],
        [49.03, 97.92, 99.17, 99.65]
    ],
}

labels = ['0', '1', '2', '3']
decoration_dict = {
    'models of different degree attribute sampling': {'marker': 's', 'c': '#3078BA'},
}
plt.rcParams['figure.figsize'] = (6.8, 3.8)
display_range_fluctuation_curve(values_dict, labels=labels, decoration_dict=decoration_dict,
                                map_title='Performance of models with varying degrees of attribute sampling.',
                                xlabel='The maximum number of masks.', ylabel='ACC (%).',
                                )  #  save_path='./savefig/Fig_10.png'

