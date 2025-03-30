from utils.calculate import calculate_cosine_similarity, calculate_reduce_dim
from utils.display import display_heatmap, display_scatter

from text2vec import SentenceModel

sentences = ['Material_plastic', 'Material_metal', 'Material_PCB',
             'Color_silver', 'Color_black', 'Color_green',
             'Geometry_camber-rectangle-border', 'Geometry_circular-column', 'Geometry_camber-rectangle',
             'Size_larger-than_PCB-cover', 'Size_smaller-than_PCB-cover_larger-than_Carrier', 'Size_smaller-than_Carrier_larger-than_Back-cover',
             'State_after_Base', 'State_after_Back-cover', 'State_after_LCD-module',
             'Front-bezel', 'Back-cover', 'Stand', 'Base', 'Circuit-boards', 'PCB-cover', 'Carrier', 'LCD-module']
model_path = "bert_uncased"
model = SentenceModel(model_path)
embeddings = model.encode(sentences)
# print(embeddings)
# print(embeddings.shape)

# Fig. 3: Cosine similarity heatmap for selected encoded characteristic and component primitives.
# The x-axis and y-axis labels follow the same order.
x = calculate_cosine_similarity(embeddings)
display_heatmap(x, labels=sentences, map_title='The cosine similarity heatmap.',
                )  # save_path='./savefig/Fig_3.png'

"""
# showing embeddings' class by PCA or TSNE is not a good method, worse much than cosine_similarity,
# because the reduction methods will ignore many dimension features, 
# but cosine_similarity products outcomes by comparing all dimensions.
"""
# y = calculate_reduce_dim(embeddings, method='TSNE')
# display_scatter(y, labels=sentences)
