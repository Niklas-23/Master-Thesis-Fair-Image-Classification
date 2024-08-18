import torch.nn as nn
import torch

from torcheval.metrics.functional import multiclass_confusion_matrix


class ImageSketchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.SPD_list = []

    def reset_spd_list(self):
        self.SPD_list = []

    def forward(self, output, target, protected_features):
        """
        Loss function copied from
        https://github.com/ubc-tea/Improving-Fairness-in-Image-Classification-via-Sketching/blob/main/face_image_classification(CelebA)/train_resnet.py
        """

        # find prediction
        _, pred = output.max(dim=1)

        binary_classification = all(target in [0, 1] for target in target) and all(a in [0, 1] for a in protected_features)

        if binary_classification:
            posz = torch.sum(protected_features)
            negz = len(protected_features) - posz

            # find num of y=1z=0 and y=1z=1
            i = 0
            y1z0 = 0
            y1z1 = 0
            for i in range(len(protected_features)):
                if pred[i] == 1 and protected_features[i] == 0:
                    y1z0 += 1
                elif pred[i] == 1 and protected_features[i] == 1:
                    y1z1 += 1

            # calculate SPD
            SPD_score = abs(y1z1 / posz - y1z0 / negz)
            self.SPD_list.append(SPD_score)

        else:
            prob_y_list = []
            for feature in torch.unique(protected_features):
                mask = protected_features == feature
                y_true_masked = torch.masked_select(target, mask)
                y_pred_masked = torch.masked_select(pred, mask)
                cnf_matrix_subset = multiclass_confusion_matrix(input=y_pred_masked, target=y_true_masked, num_classes=len(torch.unique(target)))

                fp = cnf_matrix_subset.sum(dim=0) - torch.diag(cnf_matrix_subset)
                fn = cnf_matrix_subset.sum(dim=1) - torch.diag(cnf_matrix_subset)
                tp = torch.diag(cnf_matrix_subset)
                tn = cnf_matrix_subset.sum() - (fp + fn + tp)

                positives = fp + tp
                prob_y = positives / (positives + tn + fn)

                prob_y_list.extend(prob_y)

            min_prob_y = min(prob_y_list)
            max_prob_y = max(prob_y_list)
            max_diff = abs(max_prob_y - min_prob_y)
            self.SPD_list.append(max_diff)

        # MSE of SPD in each batch
        fair_loss = torch.square(sum(self.SPD_list)) / len(self.SPD_list)

        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, target) + fair_loss

        return loss