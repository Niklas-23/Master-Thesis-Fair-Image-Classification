import collections
from typing import List

import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, \
    average_precision_score, matthews_corrcoef, precision_score, roc_auc_score

"""
For binary classification: y_true has to contain 0 and 1 values
"""


class EvaluationMetrics:
    def __init__(self, y_true, y_pred, y_prob, sensitive_features: List[int], logger, average=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.logger = logger
        self.sensitive_features = sensitive_features
        self.binary_classification = len(set(y_true)) == 2 and all(target in [0, 1] for target in y_true)
        self.average = average

        if self.binary_classification:
            self.average = "binary"
        elif average is None:
            self.average = "macro"

    def get_demographic_parity_difference(self):
        if self.binary_classification:
            return demographic_parity_difference(self.y_true, self.y_pred, sensitive_features=self.sensitive_features)

        else:
            prob_y_list = []
            for feature in np.unique(self.sensitive_features):
                fp, fn, tp, tn = self._get_masked_multi_class_positives_negatives(feature)

                positives = fp + tp
                prob_y = positives / (positives + tn + fn)

                prob_y_list.append(prob_y)

            prob_y_matrix = np.array(prob_y_list)

        min_values = np.min(prob_y_matrix, axis=0)
        max_values = np.max(prob_y_matrix, axis=0)
        max_diff_per_class = abs(max_values - min_values)  # List of length #classes (TPR per class)

        return max(max_diff_per_class), np.mean(max_diff_per_class)

    def _get_masked_multi_class_positives_negatives(self, feature):
        mask = self.sensitive_features == feature
        y_true_masked = [value for value, m in zip(self.y_true, mask) if m]
        y_pred_masked = [value for value, m in zip(self.y_pred, mask) if m]

        cnf_matrix_subset = confusion_matrix(y_true_masked, y_pred_masked, labels=np.unique(self.y_true))
        #print(cnf_matrix_subset)

        fp = cnf_matrix_subset.sum(axis=0) - np.diag(cnf_matrix_subset)
        fn = cnf_matrix_subset.sum(axis=1) - np.diag(cnf_matrix_subset)
        tp = np.diag(cnf_matrix_subset)
        tn = cnf_matrix_subset.sum() - (fp + fn + tp)
        #print("TP: ", tp)
        #print("FP: ", fp)
        #print("TN: ", tn)
        #print("FN: ", fn)
        return fp, fn, tp, tn

    def get_accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def get_f1(self):
        return f1_score(self.y_true, self.y_pred, average=self.average)

    def get_equalized_odds_difference(self):
        if self.binary_classification:
            eod = equalized_odds_difference(self.y_true, self.y_pred, sensitive_features=self.sensitive_features)
            return eod
        else:
            # Return max_eod, mean_eod_tpr, mean_eod_fpr
            return self._get_multi_class_max_eod(), self._get_multi_class_mean_eod()

    def get_strict_multiclass_equalized_odds(self):

        conditional_protected_feature_matrices = []

        for feature in np.unique(self.sensitive_features):
            mask = self.sensitive_features == feature
            y_true_masked = [value for value, m in zip(self.y_true, mask) if m]
            y_pred_masked = [value for value, m in zip(self.y_pred, mask) if m]
            cnf_matrix_subset = confusion_matrix(y_true_masked, y_pred_masked, labels=np.unique(self.y_true))
            #print(cnf_matrix_subset)
            row_sums = cnf_matrix_subset.sum(axis=1, keepdims=True)
            conditional_probabilities_subset = cnf_matrix_subset / row_sums
            conditional_probabilities_subset = np.nan_to_num(conditional_probabilities_subset, nan=0)
            conditional_protected_feature_matrices.append(conditional_probabilities_subset)
            #print(conditional_probabilities_subset)

        num_rows, num_cols = conditional_protected_feature_matrices[0].shape
        max_abs_diffs = []

        for i in range(num_rows):
            for j in range(num_cols):
                values = [matrix[i, j] for matrix in conditional_protected_feature_matrices]

                max_abs_diff = max(values) - min(values)

                max_abs_diffs.append(max_abs_diff)

        #print(max_abs_diffs)
        return max(max_abs_diffs)

    def _get_multi_class_mean_eod(self):
        max_diff_TPR_per_class, max_diff_FPR_per_class = self._get_max_multi_class_tpr_and_fpr_difference()
        TPR_mean = np.mean(max_diff_TPR_per_class)
        FPR_mean = np.mean(max_diff_FPR_per_class)

        return max(TPR_mean, FPR_mean)

    def _get_multi_class_max_eod(self):
        max_diff_TPR_per_class, max_diff_FPR_per_class = self._get_max_multi_class_tpr_and_fpr_difference()

        max_diff_tpr = max(max_diff_TPR_per_class)
        max_diff_fpr = max(max_diff_FPR_per_class)

        return max(max_diff_tpr, max_diff_fpr)

    def _get_max_multi_class_tpr_and_fpr_difference(self):
        TPR_lists = []
        FPR_lists = []

        if len(self.sensitive_features) != len(self.y_true) != self.y_pred:
            raise ValueError("List length not equal")

        class_counts = collections.Counter(self.y_pred)
        for class_type, class_count in class_counts.items():
            if class_count < 100:
                self.logger.info(f"Warning: Class {class_type} has only {class_count} predictions.")
                pass

        for feature in np.unique(self.sensitive_features):
            fp, fn, tp, tn = self._get_masked_multi_class_positives_negatives(feature)

            TPR = tp / (tp + fn)
            FPR = fp / (fp + tn)

            TPR = [0 if np.isnan(x) else x for x in TPR]  # Replace NaNs in case of zero division
            FPR = [0 if np.isnan(x) else x for x in FPR]

            TPR_lists.append(TPR)
            FPR_lists.append(FPR)

        TPR_matrix = np.array(TPR_lists)  # Matrix #sensitive attributes X #targetclasses
        #print(TPR_matrix)
        FPR_matrix = np.array(FPR_lists)  # Matrix #sensitive attributes X #targetclasses

        min_values_TPR = np.min(TPR_matrix, axis=0)
        max_values_TPR = np.max(TPR_matrix, axis=0)
        max_diff_TPR_per_class = abs(max_values_TPR - min_values_TPR)  # List of length #classes (TPR per class)

        min_values_FPR = np.min(FPR_matrix, axis=0)
        max_values_FPR = np.max(FPR_matrix, axis=0)
        max_diff_FPR_per_class = abs(max_values_FPR - min_values_FPR)  # List of length #classes (FPR per class)

        #print("MAXFDIFTPR: ", max_diff_TPR)
        #print("MAXFDIFFPR: ", max_diff_FPR)

        return max_diff_TPR_per_class, max_diff_FPR_per_class

    def get_AP(self):
        return average_precision_score(self.y_true, self.y_prob, average=self.average)

    def get_confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)

    def get_prediction_rates(self):
        if self.binary_classification:
            tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        else:
            cnf_matrix = confusion_matrix(self.y_true, self.y_pred)
            fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            tp = np.diag(cnf_matrix)
            tn = cnf_matrix.sum() - (fp + fn + tp)

            if self.average == "micro":
                tp = tp.sum()
                fn = fn.sum()
                fp = fp.sum()
                tn = tn.sum()

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = tp / (tp + fn)
        # Specificity or true negative rate
        TNR = tn / (tn + fp)
        # False positive rate
        FPR = fp / (fp + tn)
        # False negative rate
        FNR = fn / (fn + tp)

        if not self.binary_classification and self.average == "macro":
            TPR = [0 if np.isnan(x) else x for x in TPR]
            TNR = [0 if np.isnan(x) else x for x in TNR]
            FPR = [0 if np.isnan(x) else x for x in FPR]
            FNR = [0 if np.isnan(x) else x for x in FNR]

            TPR = np.mean(TPR)
            TNR = np.mean(TNR)
            FPR = np.mean(FPR)
            FNR = np.mean(FNR)

        return TPR, TNR, FPR, FNR

    def get_mcc(self):
        return matthews_corrcoef(self.y_true, self.y_pred)

    def get_precision(self):
        return precision_score(self.y_true, self.y_pred, average=self.average)

    def get_roc_auc_score(self):
        if self.binary_classification:
            return roc_auc_score(self.y_true, self.y_pred, average="macro")
        else:
            return roc_auc_score(self.y_true, self.y_prob, average="macro", multi_class='ovr')
