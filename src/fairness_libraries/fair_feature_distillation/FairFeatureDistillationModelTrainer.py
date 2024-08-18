from src.fairness_libraries.fair_feature_distillation.FairFeatureDistiallationLoss import MMDLoss, compute_hinton_loss
from src.model_training import ModelTrainer

LAMBDA_KD_STRENGTH = 0
KD_TEMPERATURE = 3
MMD_JOINT_FEATURE = False


class FairFeatureDistillationModelTrainer(ModelTrainer):
    def __init__(self, model, teacher, train_dataloader, valid_dataloader, test_dataloader, logger,
                 num_classes_target, num_classes_protected_feature,
                 device="cpu", model_path=None, lambda_feature_distill=3.0):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, logger,
                         num_classes_target,
                         device=device, model_path=model_path)
        self.teacher = teacher
        self.num_classes_protected_feature = num_classes_protected_feature
        self.distiller = MMDLoss(self.num_classes_target, self.num_classes_protected_feature, lambda_feature_distill=lambda_feature_distill)

    def train_step(self, images, targets, protected_features, epoch):
        self.optimizer.zero_grad()

        s_outputs = self.model(images, get_inter=True)
        student_logits = s_outputs[-1]

        t_outputs = self.teacher(images, get_inter=True)
        teacher_logits = t_outputs[-1]

        kd_loss = compute_hinton_loss(student_logits, t_outputs=teacher_logits,
                                      kd_temp=KD_TEMPERATURE, device=self.device) if LAMBDA_KD_STRENGTH != 0 else 0

        loss = self.loss_function(student_logits, targets.long(), protected_features)
        loss = loss + LAMBDA_KD_STRENGTH * kd_loss

        f_s = s_outputs[-2]
        f_t = t_outputs[-2]
        mmd_loss = self.distiller.forward(f_s, f_t, groups=protected_features, labels=targets, jointfeature=MMD_JOINT_FEATURE)

        train_loss = loss + mmd_loss

        train_loss.backward()
        self.optimizer.step()

        return s_outputs[-1], train_loss
