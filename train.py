'''
Standard training pipeline
'''
import os
import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
from utils import yaml_config_hook
from modules import resnet
from modules import network_sic as network
from modules import transform_sic as transform
from modules import contrastive_loss_sic as contrastive_loss
from evaluation import evaluation
from torch.utils import data
import copy
import matplotlib.pyplot as plt
import random
from utils.cluster_utils import target_distribution
from utils import kmeans
from utils import save_model
from torch.nn.functional import normalize
from utils.confusion import Confusion
from sklearn import cluster    
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

eps = 1e-8  

#Define objective function weighing scheme
ALPHA, BETA, GAMMA = 1,1,1

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def train(epoch):
    '''
        Train one epoch
    '''
    loss_epoch = 0
    for step, ((x, x_i, x_j), _) in enumerate(data_loader):
        
        optimizer.zero_grad()
       
        x = x.to('cuda')
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        
        x_i = model.get_embeddings(x_i)
        x_j = model.get_embeddings(x_j)

        #Compute Instance-wise Contrastive loss
        z_i = normalize(model.instance_projector(x_i), dim=1)
        z_j = normalize(model.instance_projector(x_j), dim=1)
        loss_instance = criterion_instance(z_i, z_j)

        #Compute Cluster loss 
        c = model.get_cluster_prob(model.get_embeddings(x))
        target = target_distribution(c).detach()
        loss_cluster = criterion_cluster((c+1e-08).log(),target)/c.shape[0]

        #Compute Anchor loss 
        p1 = model.get_cluster_prob(x_i)
        p2 = model.get_cluster_prob(x_j) 
        lds1 = criterion_anchor(p1, c)
        lds2 = criterion_anchor(p2, c)
        loss_anchor = lds1 + lds2
        
        #Compute Net loss
        loss = ALPHA * loss_instance + BETA * loss_cluster + GAMMA * loss_anchor 
        loss.backward()
        
        optimizer.step()
        
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}\t loss_anchor: {loss_anchor.item()}")

        loss_epoch += loss.item()

        # writer.add_scalar("Loss/instance_train", loss_instance.item(), epoch * len(data_loader) + step)
        # writer.add_scalar("Loss/cluster_train", loss_cluster.item(), epoch * len(data_loader) + step)
        # writer.add_scalar("Loss/consistency_train", loss_anchor.item(), epoch * len(data_loader) + step)
        # writer.add_scalar("Loss/net", loss.item(), epoch * len(data_loader) + step)

    # writer.add_scalar("Loss/train", loss_epoch, epoch)
    return loss_epoch

def evaluate(epoch, num_classes=10):
    '''
        Perform evalutation every 20 epochs
        This function computes the NMI, ACC and ARI score on 
        the learned representations as well and predicted clusters
    ''' 
    for i, (x,y) in enumerate(test_data_loader):

        with torch.no_grad():
            img = x
            img = img.to('cuda')
            embedding = model.get_embeddings(img) 
            model_prob = model.get_cluster_prob(embedding)
            label = y

            if i == 0:
                all_labels = label
                all_embeddings = embedding.detach().cpu() 
                all_prob = model_prob.detach().cpu()
            else:
                all_labels = torch.cat((all_labels, label), dim=0)
                all_embeddings = torch.cat((all_embeddings, embedding.detach().cpu()), dim=0)
                all_prob = torch.cat((all_prob, model_prob.detach().cpu()), dim=0)
            
    confusion, confusion_model = Confusion(10), Confusion(10)        
    all_pred = all_prob.max(1)[1]
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(args.num_classes)
    acc_model = confusion_model.acc()

    kmeans = cluster.KMeans(n_clusters=num_classes, random_state=args.seed)
    embeddings = all_embeddings.cpu().numpy()
    kmeans.fit(embeddings)
    pred_labels = torch.tensor(kmeans.labels_.astype(np.int))

    # clustering accuracy 
    confusion.add(pred_labels, all_labels)
    confusion.optimal_assignment(args.num_classes)
    acc = confusion.acc()
    representation_cluster_score = confusion.clusterscores()
    model_cluster_score = confusion_model.clusterscores()
    
    print('[Representation] Clustering scores:', representation_cluster_score) 
    print('[Representation] ACC: {:.3f}'.format(acc)) 

    print('[Model] Clustering scores:', model_cluster_score) 
    print('[Model] ACC: {:.3f}'.format(acc_model))  
    
    # writer.add_scalar("Representation/Accuracy/NMI", representation_cluster_score['NMI'] , epoch)
    # writer.add_scalar("Representation/Accuracy/ARI", representation_cluster_score['ARI'] , epoch)
    # writer.add_scalar("Representation/Accuracy/AMI", representation_cluster_score['AMI'] , epoch)
    # writer.add_scalar("Representation/Accuracy/ACC:", acc, epoch)
    # writer.add_scalar("Model/Accuracy/NMI", model_cluster_score['NMI'] , epoch)
    # writer.add_scalar("Model/Accuracy/ARI", model_cluster_score['ARI'] , epoch)
    # writer.add_scalar("Model/Accuracy/AMI", model_cluster_score['AMI'] , epoch)
    # writer.add_scalar("Model/Accuracy/ACC:", acc_model, epoch)

if __name__ == '__main__':   
    
    parser = {}
    config = yaml_config_hook("./config/config.yaml")
    print(config)
    for k, v in config.items():
        parser[k] = v
    args = dotdict(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data
    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(s=0.5, size=args.image_size),
        )
        class_num = 200
    else:
        raise NotImplementedError


    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    kmeans_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    if args['dataset'] == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args['dataset_dir'],
            train=True,
            download=True,
            transform=transform.Transforms(size=args['image_size']).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args['dataset_dir'],
            train=False,
            download=True,
            transform=transform.Transforms(size=args['image_size']).test_transform,
        )
        tdataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args['dataset'] == "ImageNet-10":
        tdataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args['image_size']).test_transform,
        )
        class_num = 10
    elif args['dataset'] == "ImageNet-dogs":
        tdataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs',
            transform=transform.Transforms(size=args['image_size']).test_transform,
        )
        class_num = 15
    elif args['dataset'] == "tiny-ImageNet":
        tdataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(size=args['image_size']).test_transform,
        )
        class_num = 200
    else:
        raise NotImplementedError
    
    test_data_loader = torch.utils.data.DataLoader(
        tdataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=args['workers'],
    )
    
    #Initialize backbone
    res = resnet.get_resnet(args.resnet)

    #Initalize cluster centers
    cluster_centers = kmeans.get_kmeans_center(res, kmeans_loader, 10)

    #Initialize model
    model = network.Network(res, args.feature_dim, class_num, cluster_centers)
    model.to(device)
    
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_device = torch.device("cuda")
    
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(loss_device)
    criterion_cluster = torch.nn.KLDivLoss(size_average=False).to(loss_device)
    criterion_anchor = contrastive_loss.KCL().to(loss_device)

    for epoch in range(0, 2000):

        lr = optimizer.param_groups[0]["lr"]

        #Train one epoch
        loss_epoch = train(epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")

        #Perform evaluation and save the model
        if epoch % 25 == 0:
            save_model(args, model, optimizer, epoch)
            model.eval()
            evaluate(epoch)
            model.train()
    
    save_model(args, model, optimizer, args.epochs)


