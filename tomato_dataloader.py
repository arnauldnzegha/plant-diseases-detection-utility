"""
Dataloader TomatoVillage — augmentation AGRESSIVE mais PRUDENTE (détection, format YOLO).

Entrée  : images + labels YOLO (.txt, une ligne par boîte : "class cx cy w h", normalisé 0-1).
Sortie  : images (B, 3, H, W) float [0,1] + cibles YOLO normalisées (M, 6) = [idx_batch, class, cx, cy, w, h].

Philosophie :
  AGRESSIF  -> Mosaic, MixUp, Copy-Paste ciblé sur les classes rares, géométrie + photométrie fortes,
               sur-échantillonnage des images contenant des classes rares (WeightedRandomSampler).
  PRUDENT   -> on ne détruit jamais les petites lésions :
               * filtre de taille minimale de boîte (évite le piège des boîtes < 3 px après Mosaic),
               * borne basse de l'échelle (scale_min) pour ne pas rétrécir les lésions à l'excès,
               * min_visibility : une boîte trop occultée est retirée plutôt que gardée fausse,
               * letterbox (préserve le ratio, pas de distorsion),
               * planification : Mosaic/MixUp/Copy-Paste coupés en fin d'entraînement (comme close_mosaic).

Dépendances : torch, albumentations>=1.3, opencv-python, numpy.
Remarque : certains noms d'arguments d'Albumentations varient selon la version ; testez avec la vôtre.
"""

import os
import glob
import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A


# --------------------------------------------------------------------------------------
# Configuration centrale — tous les "boutons" agressifs et prudents au même endroit.
# --------------------------------------------------------------------------------------
@dataclass
class AugConfig:
    img_size: int = 640

    # --- GARDE-FOUS PRUDENTS ------------------------------------------------------------
    min_box_size_px: float = 6.0      # toute boîte plus petite que ça (en px) est jetée.
                                      #   La zone vraiment dangereuse est < 3 px ; on garde une marge.
    min_visibility: float = 0.30      # une boîte est conservée si >= 30 % reste visible après transfo.
    scale_min: float = 0.70           # on NE rétrécit pas en dessous de 0.70 -> protège les lésions.
    scale_max: float = 1.40
    max_rotate_deg: float = 20.0      # rotation modérée (le réel ne tourne pas à 90° en général).

    # --- PROBABILITÉS AGRESSIVES --------------------------------------------------------
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.15
    copy_paste_prob: float = 0.50     # appliqué uniquement pour injecter des lésions RARES.
    hflip_prob: float = 0.20
    vflip_prob: float = 0.20

    # --- CLASSES RARES (indices YOLO) — pour Copy-Paste et pour le sampler --------------
    rare_class_ids: tuple = ()        # ex. (3, 5) si leaf miner=3 et spotted wilt=5.

    # --- PLANIFICATION ------------------------------------------------------------------
    total_epochs: int = 100
    close_aug_last_epochs: int = 10   # coupe Mosaic/MixUp/Copy-Paste sur les dernières époques.


# --------------------------------------------------------------------------------------
# Conversions de boîtes.  Format interne de travail : xyxy ABSOLU (pixels).
# --------------------------------------------------------------------------------------
def yolo_to_xyxy(boxes_n: np.ndarray, w: int, h: int) -> np.ndarray:
    """[cx,cy,bw,bh] normalisé -> [x1,y1,x2,y2] absolu."""
    if len(boxes_n) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, bw, bh = boxes_n[:, 0], boxes_n[:, 1], boxes_n[:, 2], boxes_n[:, 3]
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def xyxy_to_yolo(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """[x1,y1,x2,y2] absolu -> [cx,cy,bw,bh] normalisé."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)


def filter_small_boxes(boxes_xyxy: np.ndarray, labels: np.ndarray, min_size: float):
    """GARDE-FOU PRUDENT : retire les boîtes dont le côté le plus petit < min_size pixels."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, labels
    wbox = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    hbox = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    keep = (wbox >= min_size) & (hbox >= min_size)
    return boxes_xyxy[keep], labels[keep]


def clip_boxes(boxes_xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, w - 1)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, h - 1)
    return boxes_xyxy


# --------------------------------------------------------------------------------------
# Pipelines Albumentations.
# --------------------------------------------------------------------------------------
def build_geometric_photometric(cfg: AugConfig) -> A.Compose:
    """Pipeline complet (géométrie forte + photométrie) pour le chemin IMAGE SEULE.
    Letterbox final = préservation du ratio.  min_area dans BboxParams = garde-fou supplémentaire."""
    s = cfg.img_size
    return A.Compose(
        [
            A.Affine(
                scale=(cfg.scale_min, cfg.scale_max),
                translate_percent=(0.0, 0.10),
                rotate=(-cfg.max_rotate_deg, cfg.max_rotate_deg),
                shear=(-5, 5),
                keep_ratio=True,
                cval=114, mode=cv2.BORDER_CONSTANT,
                p=0.70,
            ),
            A.HorizontalFlip(p=cfg.hflip_prob),
            A.VerticalFlip(p=cfg.vflip_prob),
            A.RandomCrop(height=0.75*cfg.img_size, width=0.75*cfg.img_size, p=0.5),
            A.OneOf([A.RandomRotate90(p=1.0), A.Rotate(limit=90, p=1.0, border_mode=cv2.BORDER_CONSTANT)], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            # Réalisme de terrain (ombres / reflets) — faible probabilité.
            A.OneOf([A.RandomShadow(p=0.5), A.RandomSunFlare(src_radius=80, p=0.5)], p=0.10),
            A.LongestMaxSize(max_size=s),
            A.CLAHE(clip_limit=1.2, p=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=7, p=0.2),
            A.PadIfNeeded(min_height=s, min_width=s, border_mode=cv2.BORDER_CONSTANT, value=114),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=cfg.min_visibility,
            min_area=cfg.min_box_size_px ** 2,   # garde-fou : aire mini après transfo.
            clip=True,
        ),
    )


def build_photometric_only(cfg: AugConfig) -> A.Compose:
    """Photométrie seule, appliquée APRÈS Mosaic (la géométrie a déjà été lourdement sollicitée :
    enchaîner une 2e géométrie forte rétrécirait encore les lésions -> imprudent)."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=7, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.25),
        A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=5)], p=0.15),
        A.GaussNoise(var_limit=(5.0, 10.0), p=0.15),
    ])


def build_val_transform(cfg: AugConfig) -> A.Compose:
    """Validation/test : AUCUNE augmentation, seulement letterbox. (Ne jamais augmenter val/test.)"""
    s = cfg.img_size
    return A.Compose(
        [A.LongestMaxSize(max_size=s),
         A.PadIfNeeded(min_height=s, min_width=s, border_mode=cv2.BORDER_CONSTANT, value=114)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0),
    )


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class TomatoYoloDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, cfg: AugConfig, train: bool = True):
        self.cfg = cfg
        self.train = train
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.img_paths = sorted(p for e in exts for p in glob.glob(os.path.join(images_dir, e)))
        if not self.img_paths:
            raise FileNotFoundError(f"Aucune image trouvée dans {images_dir}")

        self.geo_photo = build_geometric_photometric(cfg)
        self.photo_only = build_photometric_only(cfg)
        self.val_tf = build_val_transform(cfg)

        self.epoch = 0
        self._build_rare_index()

    # -- planification : permet de couper les augmentations lourdes en fin d'entraînement ----
    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _heavy_aug_active(self) -> bool:
        if not self.train:
            return False
        return self.epoch < (self.cfg.total_epochs - self.cfg.close_aug_last_epochs)

    # -- chargement brut -------------------------------------------------------------------
    def _label_path(self, img_path: str) -> str:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.labels_dir, stem + ".txt")

    def _load(self, idx: int):
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        lp = self._label_path(self.img_paths[idx])
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            data = np.loadtxt(lp, ndmin=2).astype(np.float32)
            labels = data[:, 0].astype(np.int64)
            boxes_n = data[:, 1:5]
        else:
            labels = np.zeros((0,), dtype=np.int64)
            boxes_n = np.zeros((0, 4), dtype=np.float32)
        return img, boxes_n, labels

    # -- index des instances de classes rares (pour Copy-Paste) ----------------------------
    def _build_rare_index(self):
        self.rare_instances = []  # liste de (img_idx, boxe_yolo, label)
        if not self.cfg.rare_class_ids:
            return
        rare = set(self.cfg.rare_class_ids)
        for i, p in enumerate(self.img_paths):
            lp = self._label_path(p)
            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                data = np.loadtxt(lp, ndmin=2).astype(np.float32)
                for row in data:
                    if int(row[0]) in rare:
                        self.rare_instances.append((i, row[1:5].copy(), int(row[0])))

    # -- AUGMENTATION : Copy-Paste ciblé sur les classes rares ------------------------------
    def _copy_paste_rare(self, img, boxes_xyxy, labels):
        """Colle 1-3 lésions RARES (prélevées ailleurs) sur l'image courante.
        Ajoute directement des INSTANCES de classe rare -> attaque le déséquilibre à la source."""
        if not self.rare_instances:
            return img, boxes_xyxy, labels
        h, w = img.shape[:2]
        n_paste = random.randint(1, 3)
        new_boxes, new_labels = list(boxes_xyxy), list(labels)
        for _ in range(n_paste):
            src_idx, box_n, lbl = random.choice(self.rare_instances)
            src_img, _, _ = self._load(src_idx)
            sh, sw = src_img.shape[:2]
            sx1, sy1, sx2, sy2 = yolo_to_xyxy(box_n[None, :], sw, sh)[0].astype(int)
            sx1, sy1 = max(0, sx1), max(0, sy1)
            sx2, sy2 = min(sw, sx2), min(sh, sy2)
            if sx2 - sx1 < self.cfg.min_box_size_px or sy2 - sy1 < self.cfg.min_box_size_px:
                continue
            patch = src_img[sy1:sy2, sx1:sx2]
            ph, pw = patch.shape[:2]
            if pw >= w or ph >= h:
                continue
            # position de collage aléatoire
            px = random.randint(0, w - pw - 1)
            py = random.randint(0, h - ph - 1)
            img[py:py + ph, px:px + pw] = patch
            new_boxes.append(np.array([px, py, px + pw, py + ph], dtype=np.float32))
            new_labels.append(lbl)
        boxes_xyxy = np.array(new_boxes, dtype=np.float32) if new_boxes else np.zeros((0, 4), np.float32)
        labels = np.array(new_labels, dtype=np.int64) if new_labels else np.zeros((0,), np.int64)
        return img, boxes_xyxy, labels

    # -- AUGMENTATION : Mosaic (4 images dans une sortie de taille img_size) ----------------
    def _mosaic(self, idx: int):
        """Combine 4 images en quadrants. Les lésions se retrouvent ~2x plus petites ->
        le filtre min_box_size_px (appliqué ensuite) est le garde-fou contre les boîtes inapprenables."""
        s = self.cfg.img_size
        half = s // 2
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        idxs = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        # offsets des 4 quadrants
        quad = [(0, 0), (half, 0), (0, half), (half, half)]
        all_boxes, all_labels = [], []
        for (ox, oy), k in zip(quad, idxs):
            im, boxes_n, labels = self._load(k)
            ih, iw = im.shape[:2]
            scale = min(half / iw, half / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            im_r = cv2.resize(im, (nw, nh))
            canvas[oy:oy + nh, ox:ox + nw] = im_r
            if len(boxes_n):
                b = yolo_to_xyxy(boxes_n, iw, ih) * scale
                b[:, [0, 2]] += ox
                b[:, [1, 3]] += oy
                all_boxes.append(b)
                all_labels.append(labels)
        if all_boxes:
            boxes_xyxy = np.concatenate(all_boxes, 0)
            labels = np.concatenate(all_labels, 0)
        else:
            boxes_xyxy = np.zeros((0, 4), np.float32)
            labels = np.zeros((0,), np.int64)
        boxes_xyxy = clip_boxes(boxes_xyxy, s, s)
        boxes_xyxy, labels = filter_small_boxes(boxes_xyxy, labels, self.cfg.min_box_size_px)
        return canvas, boxes_xyxy, labels

    # -- AUGMENTATION : MixUp --------------------------------------------------------------
    def _mixup(self, img_a, boxes_a, labels_a, img_b, boxes_b, labels_b):
        lam = np.random.beta(8.0, 8.0)  # mélange doux, centré sur 0.5 -> prudent
        mixed = (lam * img_a.astype(np.float32) + (1 - lam) * img_b.astype(np.float32)).astype(np.uint8)
        boxes = np.concatenate([boxes_a, boxes_b], 0) if len(boxes_a) or len(boxes_b) else np.zeros((0, 4), np.float32)
        labels = np.concatenate([labels_a, labels_b], 0) if len(labels_a) or len(labels_b) else np.zeros((0,), np.int64)
        return mixed, boxes, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        cfg = self.cfg
        s = cfg.img_size
        heavy = self._heavy_aug_active()

        if not self.train:
            # --- VALIDATION / TEST : letterbox seul, zéro augmentation ---
            img, boxes_n, labels = self._load(idx)
            out = self.val_tf(image=img, bboxes=boxes_n.tolist(), class_labels=labels.tolist())
            img = out["image"]
            boxes_n = np.array(out["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.array(out["class_labels"], dtype=np.int64)
            return self._to_tensor(img, boxes_n, labels)

        # ---------------- CHEMIN D'ENTRAÎNEMENT ----------------
        if heavy and random.random() < cfg.mosaic_prob:
            # --- Mosaic puis photométrie seule (pas de 2e géométrie) ---
            img, boxes_xyxy, labels = self._mosaic(idx)
            img = self.photo_only(image=img)["image"]
            boxes_n = xyxy_to_yolo(boxes_xyxy, s, s)
        else:
            # --- Image seule : (Copy-Paste rare) -> géométrie + photométrie complètes ---
            img, boxes_n, labels = self._load(idx)
            h, w = img.shape[:2]
            boxes_xyxy = yolo_to_xyxy(boxes_n, w, h)
            if heavy and self.rare_instances and random.random() < cfg.copy_paste_prob:
                img, boxes_xyxy, labels = self._copy_paste_rare(img, boxes_xyxy, labels)
            boxes_n = xyxy_to_yolo(boxes_xyxy, img.shape[1], img.shape[0])
            out = self.geo_photo(image=img, bboxes=boxes_n.tolist(), class_labels=labels.tolist())
            img = out["image"]
            boxes_n = np.array(out["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.array(out["class_labels"], dtype=np.int64)

        # --- MixUp éventuel (sur le résultat précédent) ---
        if heavy and random.random() < cfg.mixup_prob:
            jdx = random.randint(0, len(self) - 1)
            img_b, bn_b, lb_b = self._mosaic(jdx) if random.random() < 0.5 else self._load(jdx)
            if img_b.shape[:2] != (s, s):
                img_b = cv2.resize(img_b, (s, s))
                bn_b = xyxy_to_yolo(yolo_to_xyxy(bn_b, img_b.shape[1], img_b.shape[0]), s, s) if len(bn_b) else bn_b
            ba = yolo_to_xyxy(boxes_n, s, s)
            bb = bn_b if bn_b.ndim == 2 and bn_b.shape[1] == 4 else np.zeros((0, 4), np.float32)
            bb = yolo_to_xyxy(bb, s, s) if len(bb) else bb
            img, mb, labels = self._mixup(img, ba, labels, img_b, bb, lb_b)
            boxes_n = xyxy_to_yolo(mb, s, s)

        # --- GARDE-FOU FINAL : on rejette toute boîte trop petite, quel que soit le chemin ---
        boxes_xyxy = yolo_to_xyxy(boxes_n, s, s)
        boxes_xyxy, labels = filter_small_boxes(boxes_xyxy, labels, cfg.min_box_size_px)
        boxes_n = xyxy_to_yolo(boxes_xyxy, s, s)
        return self._to_tensor(img, boxes_n, labels)

    def _to_tensor(self, img, boxes_n, labels):
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        if len(boxes_n) == 0:
            target = torch.zeros((0, 5), dtype=torch.float32)
        else:
            target = torch.cat(
                [torch.from_numpy(labels.astype(np.float32)).unsqueeze(1),
                 torch.from_numpy(boxes_n.astype(np.float32))], dim=1)  # (N, 5) = class, cx, cy, w, h
        return img_t, target


# --------------------------------------------------------------------------------------
# collate : empile les images, préfixe l'index de batch aux cibles (convention YOLO).
# --------------------------------------------------------------------------------------
def yolo_collate(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    out = []
    for i, t in enumerate(targets):
        if t.numel():
            idx_col = torch.full((t.shape[0], 1), float(i))
            out.append(torch.cat([idx_col, t], dim=1))  # (N, 6) = [batch_idx, class, cx, cy, w, h]
    targets = torch.cat(out, 0) if out else torch.zeros((0, 6), dtype=torch.float32)
    return imgs, targets


# --------------------------------------------------------------------------------------
# Sampler pondéré : sur-échantillonne les images contenant des classes RARES.
# --------------------------------------------------------------------------------------
def build_weighted_sampler(dataset: TomatoYoloDataset, cfg: AugConfig) -> WeightedRandomSampler:
    # fréquence d'instances par classe
    class_count = {}
    img_classes = []
    for p in dataset.img_paths:
        lp = dataset._label_path(p)
        cls_here = set()
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            data = np.loadtxt(lp, ndmin=2).astype(np.float32)
            for row in data:
                c = int(row[0])
                class_count[c] = class_count.get(c, 0) + 1
                cls_here.add(c)
        img_classes.append(cls_here)
    # poids image = max des poids inverse-fréquence de ses classes (l'image "vaut" sa classe la plus rare)
    inv = {c: 1.0 / math.sqrt(n) for c, n in class_count.items()}  # sqrt -> sur-échantillonnage moins brutal
    weights = []
    for cls_here in img_classes:
        weights.append(max((inv[c] for c in cls_here), default=min(inv.values()) if inv else 1.0))
    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloader(images_dir, labels_dir, cfg: AugConfig, train=True, batch_size=16, num_workers=4):
    ds = TomatoYoloDataset(images_dir, labels_dir, cfg, train=train)
    sampler = build_weighted_sampler(ds, cfg) if (train and cfg.rare_class_ids) else None
    loader = DataLoader(
        ds, batch_size=batch_size,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        collate_fn=yolo_collate, drop_last=train,
    )
    return ds, loader


# --------------------------------------------------------------------------------------
# Démo / contrôle de cohérence.
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = AugConfig(
        img_size=640,
        rare_class_ids=(3, 5),     # <-- À ADAPTER : mettez ici les indices de vos classes rares.
        total_epochs=100,
        close_aug_last_epochs=10,
    )
    ds, loader = build_dataloader(
        images_dir="TomatoVillage/detection/train/images",
        labels_dir="TomatoVillage/detection/train/labels",
        cfg=cfg, train=True, batch_size=8, num_workers=0,
    )
    ds.set_epoch(0)
    imgs, targets = next(iter(loader))
    print("images :", imgs.shape)                 # (8, 3, 640, 640)
    print("cibles :", targets.shape)              # (M, 6) = [batch_idx, class, cx, cy, w, h]
    print("classes présentes :", targets[:, 1].unique().tolist() if targets.numel() else "aucune")
