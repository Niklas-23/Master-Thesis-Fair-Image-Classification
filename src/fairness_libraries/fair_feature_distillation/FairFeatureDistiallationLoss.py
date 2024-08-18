import torch
import torch.nn.functional as F
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self, num_classes_target, num_classes_protected_feature, kernel="rbf", lambda_feature_distill=3.0, sigma=1.0,):
        super(MMDLoss, self).__init__()
        self.lambda_feature_distill = lambda_feature_distill
        self.sigma = sigma
        self.num_groups = num_classes_protected_feature
        self.num_classes = num_classes_target
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels, jointfeature=False):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        if jointfeature:
            K_TS, sigma_avg = self.pdist(teacher, student,
                              sigma_base=self.sigma, kernel=self.kernel)
            K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
            K_SS, _ = self.pdist(student, student,
                              sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        else:
            with torch.no_grad():
                _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

            for c in range(self.num_classes):
                if len(teacher[labels==c]) == 0:
                    continue
                for g in range(self.num_groups):
                    if len(student[(labels==c) * (groups == g)]) == 0:
                        continue
                    K_TS, _ = self.pdist(teacher[labels == c], student[(labels == c) * (groups == g)],
                                                 sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                    K_SS, _ = self.pdist(student[(labels == c) * (groups == g)], student[(labels == c) * (groups == g)],
                                         sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                    K_TT, _ = self.pdist(teacher[labels == c], teacher[labels == c], sigma_base=self.sigma,
                                         sigma_avg=sigma_avg, kernel=self.kernel)

                    mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        loss = (1/2) * self.lambda_feature_distill * mmd_loss

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg


def compute_hinton_loss(outputs, t_outputs=None, teacher=None, t_inputs=None, kd_temp=3, device=0):
    if t_outputs is None:
        if (t_inputs is not None and teacher is not None):
            t_outputs = teacher(t_inputs)
        else:
            Exception('Nothing is given to compute hinton loss')

    soft_label = F.softmax(t_outputs / kd_temp, dim=1).to(device).detach()
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / kd_temp, dim=1),
                                                  soft_label) * (kd_temp * kd_temp)

    return kd_loss