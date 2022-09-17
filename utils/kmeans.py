import torch
import numpy as np
from sklearn.cluster import KMeans

def get_kmeans_center(model, dataloader, num_classes):
    
    model.to('cuda')
    
    for i, (x,y) in enumerate(dataloader):
        
        img = x[0]
        img = img.to('cuda')
        embedding = model(img) 
        label = y
        
        if i == 0:
#             all_labels = label
            all_embeddings = embedding.detach().cpu() 
        else:
#             all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = torch.cat((all_embeddings, embedding.detach().cpu()), dim=0)
            
    kmeans = KMeans(n_clusters=num_classes, random_state=0)
    all_embeddings = all_embeddings.numpy()
    print(all_embeddings.shape)
    kmeans.fit(all_embeddings)
    
    cluster_centers = torch.Tensor(kmeans.cluster_centers_).to('cuda')
    return cluster_centers
