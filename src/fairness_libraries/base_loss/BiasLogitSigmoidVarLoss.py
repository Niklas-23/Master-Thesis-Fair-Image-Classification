import torch.nn as nn
import torch.nn.functional as F
import torch


class BiasLogitSigmoidVarLoss(nn.Module):
    def __init__(self, var_gamma=1, sigmoid_scale=1, num_classes=2):
        super().__init__()

        self.var_gamma = var_gamma
        self.sigmoid_scale = sigmoid_scale
        self.num_classes = num_classes

    def _get_bias_acc_mean(self, bias_losses):
        return torch.mean(bias_losses)

    def forward(self,
        target_logits,
        target_labels,
        protected_features
    ):
        #print(target_labels)
        #print("target_logits")
        #print(target_logits)

        ce_loss = F.cross_entropy(target_logits, target_labels)

        bias_losses = []
        bias_classes = protected_features.unique()

        try:
            correct_mask = F.one_hot(target_labels.to(torch.int64), num_classes=self.num_classes).bool()
            correct_logits = target_logits[correct_mask]

            incorrect_logits = target_logits[~correct_mask].view(target_logits.shape[0], -1)

        except Exception as e:
            print(e)
            print(target_labels)
            print("target_logits")
            print(target_logits)


        try:
            highest_incorrect_logit, _ = torch.max(incorrect_logits, dim=1)
            loss_diff = correct_logits - highest_incorrect_logit # 2 -32

            sig_loss = torch.sigmoid(loss_diff*self.sigmoid_scale)

        except Exception as e:
            print("Correct mask")
            print(correct_mask)
            print("Correct logits")
            print(correct_logits)
            print("INCORRECT MASK")
            print(~correct_mask)
            print("incorrect_logits")
            print(incorrect_logits)

        for bias_class in bias_classes:
            bias_mask = bias_class == protected_features
            bias_loss = sig_loss[bias_mask].mean()
            bias_losses.append(bias_loss)

        bias_losses = torch.stack(bias_losses)

        mean_loss = self._get_bias_acc_mean(bias_losses)
        mean_var = torch.mean((bias_losses - mean_loss) ** 2)

        loss = ce_loss + self.var_gamma * mean_var

        return loss
