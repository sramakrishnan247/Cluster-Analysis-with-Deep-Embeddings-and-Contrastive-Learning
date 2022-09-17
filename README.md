# Cluster Analysis with Deep Embeddings and Contrastive Learning
### Paper: https://arxiv.org/abs/2109.12714

#### Summary:
Unsupervised disentangled representation learning is a long-standing problem in computer vision. This work proposes a novel framework for performing image clustering from deep embeddings by combining instance-level contrastive learning with a deep embedding based cluster center predictor. Our approach jointly learns representations and predicts cluster centers in an end-to-end manner. This is accomplished via a three-pronged approach that combines a clustering loss, an instance-wise contrastive loss, and an anchor loss. Our fundamental intuition is that using an ensemble loss that incorporates instance-level features and a clustering procedure focusing on semantic similarity reinforces learning better representations in the latent space. We observe that our method performs exceptionally well on popular vision datasets when evaluated using standard clustering metrics such as Normalized Mutual Information (NMI), in addition to producing geometrically well-separated cluster embeddings as defined by the Euclidean distance. Our framework performs on par with widely accepted clustering methods and outperforms the state-of-the-art contrastive learning method on the CIFAR-10 dataset with an NMI score of 0.772, a 7-8% improvement on the strong baseline.

#### Datasets:
CIFAR-10 will be automatically downloaded. \
For ImageNet-10, ImageNet-dogs copy the repsective classes as given in info.txt. \
Tiny-ImageNet - Download the dataset from here: http://cs231n.stanford.edu/tiny-imagenet-200.zip

#### Requirements:
 - python>=3.8 \
 - pytorch>=1.6.0 \
 - torchvision>=0.8.1 \
 - munkres>=1.1.4 \
 - numpy>=1.19.2 \
 - opencv-python>=4.4.0.46 \
 - pyyaml>=5.3.1 \
 - scikit-learn>=0.23.2 \
 - cudatoolkit>=11.0

#### Configuration
    config/config.yaml

#### Pretraining: 
    ./run.sh
    
#### Cite:
    @article{DBLP:journals/corr/abs-2109-12714,
    author    = {Ramakrishnan Sundareswaran and
                 Jansel Herrera{-}Gerena and
                 John Just and
                 Ali Janessari},
    title     = {Cluster Analysis with Deep Embeddings and Contrastive Learning},
    journal   = {CoRR},
    volume    = {abs/2109.12714},
    year      = {2021},
    url       = {https://arxiv.org/abs/2109.12714},
    eprinttype = {arXiv},
    eprint    = {2109.12714},
    timestamp = {Mon, 04 Oct 2021 17:22:25 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2109-12714.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
    }
