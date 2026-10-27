from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO
import torch
import torch.nn.functional as F

class KDTrainer(DetectionTrainer):
    
    def set_teachers(self, t1, t2, temperature, alpha):
        self.teacher1    = t1
        self.teacher2    = t2
        self.temperature = temperature
        self.alpha       = alpha
        self.kd_loss_history = []

    def get_teacher_soft_logits(self, imgs):
        """Moyenne softmax softened des deux teachers."""
        def extract(model, imgs):
            preds = model.model(imgs)
            cls_list = []
            if isinstance(preds, (list, tuple)):
                for p in preds:
                    if isinstance(p, torch.Tensor) and p.ndim == 3:
                        nc = p.shape[1] - 4
                        if nc > 0:
                            cls_list.append(p[:, 4:, :].mean(dim=2))
            if cls_list:
                return torch.cat(cls_list, dim=1)
            return None

        with torch.no_grad():
            l1 = extract(self.teacher1, imgs)
            l2 = extract(self.teacher2, imgs)

        if l1 is None or l2 is None:
            return None

        min_dim = min(l1.shape[1], l2.shape[1])
        p1 = F.softmax(l1[:, :min_dim] / self.temperature, dim=1)
        p2 = F.softmax(l2[:, :min_dim] / self.temperature, dim=1)
        return (p1 + p2) / 2.0

    def loss_items_to_loss(self, preds, batch):
        """
        Override : calcule la loss YOLO standard + KD loss.
        Appelé AVANT optimizer.step().
        """
        # Loss YOLO standard
        loss, loss_items = super().loss_items_to_loss(preds, batch)
        # ── Diagnostic ──────────────────────────────
        print(f"[KDTrainer] loss_items_to_loss appelé ✅")
        print(f"[KDTrainer] loss avant KD : {loss.item():.4f}")
        # ─────────────────────────────────────────────
        # KD loss
        imgs = batch['img'].to(self.device)
        teacher_probs = self.get_teacher_soft_logits(imgs)

        if teacher_probs is not None:
            # Extraire les logits student depuis preds
            student_cls_list = []
            if isinstance(preds, (list, tuple)):
                for p in preds:
                    if isinstance(p, torch.Tensor) and p.ndim == 3:
                        nc = p.shape[1] - 4
                        if nc > 0:
                            student_cls_list.append(p[:, 4:, :].mean(dim=2))

            if student_cls_list:
                student_cls = torch.cat(student_cls_list, dim=1)
                min_dim     = min(student_cls.shape[1], teacher_probs.shape[1])
                s_log       = F.log_softmax(
                    student_cls[:, :min_dim] / self.temperature, dim=1
                )
                t_prob      = teacher_probs[:, :min_dim]
                kd_loss     = (
                    F.kl_div(s_log, t_prob, reduction='batchmean')
                    * (self.temperature ** 2)
                    * self.alpha
                )
                loss = loss + kd_loss
                self.kd_loss_history.append(kd_loss.item())

        return loss, loss_items