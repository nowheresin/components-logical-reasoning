from matplotlib import pyplot as plt
from utils.display import display_range_fluctuation_curve, display_multi_bar

# Fig. 4: Distribution of characteristics for misclassified components using the common logical model.
# multi_bar
values_dict = {
    'Front bezel': [[0.71, 0.68, 0.72, 0.71, 0.67], [0.68, 0.8, 0.66, 0.75, 0.7], [0.86, 0.62, 0.7, 0.64, 0.72], [0.76, 0.7, 0.78, 0.58, 0.77], [0.7, 0.69, 0.73, 0.72, 0.7]],
    'Back cover': [[0.74, 0.72, 0.75, 0.76, 0.72], [0.69, 0.69, 0.73, 0.72, 0.68], [0.87, 0.58, 0.75, 0.87, 0.71], [0.79, 0.75, 0.57, 0.79, 0.81], [0.74, 0.71, 0.69, 0.66, 0.71]],
    'Stand': [[0.78, 0.77, 0.75, 0.76, 0.76], [0.73, 0.75, 0.75, 0.72, 0.75], [0.75, 0.75, 0.74, 0.7, 0.69], [0.81, 0.87, 0.8, 0.92, 0.8], [0.62, 0.57, 0.62, 0.57, 0.63]],
    'Base': [[0.69, 0.74, 0.76, 0.52, 0.62], [0.7, 0.69, 0.71, 0.7, 0.68], [0.69, 0.68, 0.76, 0.65, 0.68], [0.77, 0.89, 1.0, 0.78, 0.77], [0.67, 1.0, 1.0, 0.55, 0.31]],
    'Circuit board': [[0.77, 0.78, 0.78, 0.83, 0.84], [0.7, 0.67, 0.63, 0.74, 0.7], [0.7, 0.68, 0.7, 0.7, 0.69], [0.7, 0.66, 0.85, 0.66, 0.62], [0.81, 0.83, 0.74, 0.71, 0.83]],
    'PCB cover': [[0.7, 0.69, 0.73, 0.69, 0.69], [0.76, 0.77, 0.68, 0.55, 0.58], [0.93, 0.58, 0.69, 0.65, 0.7], [0.66, 0.71, 0.69, 0.7, 0.73], [0.75, 0.65, 0.68, 0.65, 0.69]],
    'Carrier': [[0.7, 0.7, 0.71, 0.69, 0.71], [0.68, 0.7, 0.7, 0.72, 0.73], [0.63, 0.69, 0.66, 0.7, 0.79], [0.72, 0.66, 0.59, 0.7, 0.73], [0.71, 0.71, 0.69, 0.7, 0.71]],
    'LCD module': [[0.71, 0.72, 0.69, 0.7, 0.69], [0.69, 0.7, 0.72, 0.7, 0.68], [0.76, 0.76, 0.88, 0.66, 0.65], [0.72, 0.69, 0.73, 0.7, 0.69], [0.72, 0.67, 0.67, 0.69, 0.73]],
}
labels = ['Material', 'Color', 'Geometry', 'Size', 'State']
plt.rcParams['figure.figsize'] = (12.8, 6.3)  #(9.8, 4.8)
display_multi_bar(values_dict, labels=labels, type='var',
                  map_title='Distribution of characteristics for misclassified components using the common logical model.',
                  xlabel='The component classes.', ylabel='Frequency.',
                  )  # save_path='./savefig/Fig_4.png'

# Fig. 5: Distribution of characteristics for misclassified components using the sampling logical model.
# multi_bar
values_dict = {
    'Front bezel': [[0.73, 1.0, 0.95, 0.0, 0.8], [0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.47, 0.53, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
    'Back cover': [[1.0, 1.0, 1.0, 0.44, 0.0], [0.79, 0.77, 0.82, 0.96, 0.76], [0.86, 0.89, 0.92, 0.59, 0.0], [0.76, 0.91, 0.74, 1.0, 0.0], [0.88, 0.8, 0.8, 0.87, 0.66]],
    'Stand': [[0.82, 0.77, 0.77, 0.92, 0.65], [1.0, 1.0, 0.0, 1.0, 0.0], [0.79, 0.64, 0.56, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0, 0.0]],
}
labels = ['Material', 'Color', 'Geometry', 'Size', 'State']
plt.rcParams['figure.figsize'] = (12.8, 6.3)  #(9.8, 4.8)
display_multi_bar(values_dict, labels=labels, type='var',
                  map_title='Distribution of characteristics for misclassified components using the sampling logical model.',
                  xlabel='The component classes.', ylabel='Frequency.',
                  )  # save_path='./savefig/Fig_5.png'

# Fig. 6: Comparison of classification accuracy for the common logic model (common) and
# the sampling logic model (sampling) with and without domain-specific data processing.
# multi_bar
values_dict = {
    'common': [[45.08, 39.64], [38.36, 39.26], [43.60, 44.76], [43.21, 46.54], [41.74, 47.99]],
    'sampling': [[92.42, 92.85], [92.93, 95.99], [89.33, 97.04], [92.81, 98.21], [89.15, 95.96]],
}
labels = ['w/o data processing', 'w/ data processing']
plt.rcParams['figure.figsize'] = (5.4, 3.8)  #(9.8, 4.8)
display_multi_bar(values_dict, labels=labels, type='std',
                  map_title='The distribution.',
                  xlabel='Type of model.', ylabel='ACC.',
                  )  # save_path='./savefig/Fig_6.png'

# Fig. 7: Performance of models with varying degrees of characteristic sampling.
# range_fluctuation_curve
values_dict = {
    'models of different degree characteristic sampling':[
        [39.64, 87.31, 92.85, 96.75, 96.24],
        [39.26, 95.29, 95.99, 94.82, 98.85],
        [44.76, 75.72, 97.04, 98.58, 98.78],
        [46.54, 73.31, 98.21, 98.39, 98.38],
        [47.99, 96.51, 95.96, 98.43, 98.61]
    ],
}
labels = ['0', '1', '2', '3', '4']
decoration_dict = {
    'models of different degree characteristic sampling': {'marker': 's', 'c': '#3078BA'},
}
plt.rcParams['figure.figsize'] = (6.8, 3.8)
display_range_fluctuation_curve(values_dict, labels=labels, decoration_dict=decoration_dict,
                                map_title='Performance of models with varying degrees of characteristic sampling.',
                                xlabel='The maximum number of masks.', ylabel='ACC.',
                                )  # save_path='./savefig/Fig_7.png'

