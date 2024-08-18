import numpy as np


def get_feature_list_from_dataloader(dataloader):
    x = []
    y_true = []
    protected_features = []
    for images, targets, prot_features in dataloader:
        x.extend(images)
        y_true.extend(targets)
        protected_features.extend(prot_features)

    print(np.array(x).shape)

    return x, y_true, protected_features
