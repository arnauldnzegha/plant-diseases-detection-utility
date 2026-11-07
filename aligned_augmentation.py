from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import albumentations as A
import cv2
import numpy as np

TRANSFORM_NAMES = (
    "rgb_shift",
    "brightness_contrast",
    "rotate",
    "random_rotate90",
    "horizontal_flip",
    "vertical_flip",
    "random_crop",
)
GEOMETRIC = ("rotate", "random_rotate90", "horizontal_flip", "vertical_flip", "random_crop")
# swap_classes : pour les images de carence, couleur -> transpositions de D4 (symétries diagonales)
COLOR_TO_TRANSPOSE = {"rgb_shift": "transpose", "brightness_contrast": "anti_transpose"}
# swap_mode="illumination_cycle" : couleur -> éclairage local, puis cycle de 10 avec transpositions + combinaison
COLOR_TO_ILLUMINATION = {"rgb_shift": "shadow", "brightness_contrast": "sunflare"}
ILLUMINATION_CYCLE_EXTRA = ("transpose", "anti_transpose", "tcc")
# swap_mode="transpose_cycle" : après les 7 du naïf, une fois ces 4 transformations, puis les 5 géométriques
TRANSPOSE_CYCLE_ONCE = ("transpose_rgb_shift", "anti_transpose_bc")
COLOR = ("rgb_shift", "brightness_contrast")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
N_SAME_TRANSFORM_RETRIES = 3
EXTRA_D4_AFFINE = "d4_affine"   # transformation de complément (extra_mode="d4_affine")


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
@dataclass
class AlignedAugConfig:
    images_dir: str
    labels_dir: str
    out_images_dir: str
    out_labels_dir: str

    alpha: float = 0.0                    # 0 = naïf ; > 0 = class-aware
    budget: int | None = None             # None -> 7 × nombre d'images (budget de la DA naïve)
    k_max: int = 28                       # plafond de copies par image source (ignoré si alpha = 0)
    boost_classes: tuple = ()             # classes ciblées (ids ou noms) ; () = toutes les classes
    class_names: dict = field(default_factory=dict)  # {id: nom}, pour les rapports et boost_classes

    combine: bool = False                 # ablation : + 1 transformation de l'autre famille par copie
    combine_from_rep: int = 1             # 1 = combiner seulement à partir de la 8e copie
    combine_exclude: tuple = ("rotate",)  # transformations jamais combinées (ni principale ni secondaire)
    extra_mode: str = "repeat"            # copies au-delà de 7 : "repeat" (cycle des 7) ou "d4_affine"
    swap_classes: tuple = ()              # images contenant ces classes : catalogue modifié (swap_mode)
                                          # ex. (4, 5, 6) = Mg, N, K. () = désactivé
    swap_mode: str = "transpose"          # "transpose" ou "illumination_cycle"
    shadow_intensity: tuple = (0.2, 0.35) # illumination_cycle : assombrissement de l'ombre (20 à 35 %)
    sunflare_radius_frac: float = 0.125   # illumination_cycle : rayon du reflet / grand côté (80 px à 640)
    tcc_rgb_shift: int = 10               # illumination_cycle : RGBShift ±10 dans la combinaison
    tc_rgb_shift: int = 10                # transpose_cycle : RGBShift ±10 avec la transposition
    tac_bc_limit: float = 0.1             # transpose_cycle : luminosité/contraste ±0,1 avec l'anti-transposition
    affine_rotate: float = 5.0            # d4_affine : rotation ±5° (largest_box : aire des boîtes ≤ ×1,17)
    affine_scale: tuple = (0.9, 1.1)      # d4_affine : échelle (boîtes exactes)
    affine_shear: float = 0.0             # d4_affine : cisaillement (0 = désactivé ; 5° -> aire ×1,18)
    affine_rotate_method: str = "largest_box"  # "largest_box" : ne coupe jamais l'objet (boîte plus lâche)
                                               # "ellipse" : boîte serrée mais suppose un objet elliptique
    # copy-paste léger (extra_mode="d4_affine_cp") — uniquement sur les copies d4_affine
    cp_prob: float = 0.5                  # probabilité de coller sur une copie d4_affine
    cp_max_patches: int = 1               # lésions collées au plus par copie
    cp_max_rel_size: float = 0.25         # côté de la lésion collée ≤ 25 % du côté de l'image
    cp_feather: float = 0.15              # largeur du fondu des bords (fraction de la taille du patch)
    cp_margin: float = 0.02               # écart minimal aux boîtes existantes (fraction du grand côté)
    cp_position_tries: int = 30           # essais de position avant d'abandonner le collage
    copy_originals: bool = True
    dedupe: bool = False
    retry_failed: bool = False
    min_visibility: float = 0.2
    crop_fraction: float = 0.7
    seed: int = 42
    jpeg_quality: int = 95                # qualité par défaut de cv2.imwrite


# ----------------------------------------------------------------------------------------------
# Transformations : UNE définition, utilisée par les deux méthodes
# ----------------------------------------------------------------------------------------------
def _bbox_params(min_visibility: float) -> A.BboxParams:
    return A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=min_visibility)


def build_transform(names, img_h: int, img_w: int, cfg: AlignedAugConfig) -> A.Compose:
    """names : une transformation (str) ou plusieurs (tuple). Couleur appliquée avant la géométrie."""
    if isinstance(names, str):
        names = (names,)
    ordered = [n for n in names if n in COLOR] + [n for n in names if n not in COLOR]
    flat = []
    for n in ordered:  # liste à plat : un Sequential imbriqué ne reçoit pas la graine de Compose
        t = _single_transform(n, img_h, img_w, cfg)
        flat.extend(t if isinstance(t, list) else [t])
    return A.Compose(flat, bbox_params=_bbox_params(cfg.min_visibility))


def secondary_transform(stem: str, name: str, rep: int, attempt: int, cfg) -> str | None:
    """Transformation secondaire (mode combiné), tirée dans l'autre famille de façon déterministe.
    None si la copie ne doit pas être combinée (répétition trop basse ou transformation exclue)."""
    if (not cfg.combine or rep < cfg.combine_from_rep or name in cfg.combine_exclude
            or name == EXTRA_D4_AFFINE):
        return None
    family = [t for t in (GEOMETRIC if name in COLOR else COLOR) if t not in cfg.combine_exclude]
    if not family:
        return None
    return random.Random(sample_seed(cfg.seed, stem, name + "|secondary", rep, attempt)).choice(family)


def _single_transform(name: str, img_h: int, img_w: int, cfg: AlignedAugConfig):
    if name == "rgb_shift":
        t = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)
    elif name == "brightness_contrast":
        t = A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
    elif name == "rotate":
        t = A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=1.0)
    elif name == "random_rotate90":
        t = A.RandomRotate90(p=1.0)
    elif name == "horizontal_flip":
        t = A.HorizontalFlip(p=1.0)
    elif name == "vertical_flip":
        t = A.VerticalFlip(p=1.0)
    elif name == "random_crop":
        t = A.RandomCrop(height=int(img_h * cfg.crop_fraction), width=int(img_w * cfg.crop_fraction), p=1.0)
    elif name == "transpose":            # symétrie sur la diagonale principale
        t = A.Transpose(p=1.0)
    elif name == "anti_transpose":       # symétrie sur l'anti-diagonale = transposition puis rotation de 180°
        t = [A.Transpose(p=1.0), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]
    elif name == "shadow":               # une ombre modérée, n'importe où dans l'image
        t = A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 1),
                           shadow_intensity_range=cfg.shadow_intensity, p=1.0)
    elif name == "sunflare":             # reflet, rayon proportionnel à la taille de l'image
        t = A.RandomSunFlare(flare_roi=(0, 0, 1, 1),
                             src_radius=max(1, int(round(cfg.sunflare_radius_frac * max(img_h, img_w)))), p=1.0)
    elif name == "transpose_rgb_shift":  # couleur ±tc_rgb_shift puis transposition
        s = cfg.tc_rgb_shift
        t = [A.RGBShift(r_shift_limit=s, g_shift_limit=s, b_shift_limit=s, p=1.0), A.Transpose(p=1.0)]
    elif name == "anti_transpose_bc":    # luminosité/contraste ±tac_bc_limit puis anti-transposition
        b = cfg.tac_bc_limit
        t = [A.RandomBrightnessContrast(brightness_limit=b, contrast_limit=b, p=1.0),
             A.Transpose(p=1.0), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]
    elif name in ("tcc_transpose", "tcc_anti_transpose"):
        # combinaison : couleur ±tcc_rgb_shift, recadrage 70 %, puis transposition
        # (recadrage AVANT la transposition : ses dimensions sont celles de l'image d'origine)
        s = cfg.tcc_rgb_shift
        t = [A.RGBShift(r_shift_limit=s, g_shift_limit=s, b_shift_limit=s, p=1.0),
             A.RandomCrop(height=int(img_h * cfg.crop_fraction), width=int(img_w * cfg.crop_fraction), p=1.0),
             A.Transpose(p=1.0)]
        if name == "tcc_anti_transpose":
            t += [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]
    elif name == EXTRA_D4_AFFINE:
        # D4 (8 symétries du carré, boîtes exactes) puis Affine léger, sans couleur.
        # largest_box (défaut) : la boîte englobe toujours tout l'objet tourné (un peu plus lâche).
        t = [
            A.D4(p=1.0),
            A.Affine(rotate=(-cfg.affine_rotate, cfg.affine_rotate), scale=cfg.affine_scale,
                     shear=(-cfg.affine_shear, cfg.affine_shear), rotate_method=cfg.affine_rotate_method,
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
        ]
    else:
        raise ValueError(f"Transformation inconnue : {name}")
    return t


def sample_seed(seed: int, stem: str, name: str, rep: int, attempt: int) -> int:
    """Graine déterministe par échantillon : la même (image, transformation, répétition, essai)
    donne la même copie quelle que soit la méthode ou l'ordre de génération."""
    key = f"{seed}|{stem}|{name}|{rep}|{attempt}".encode()
    return int.from_bytes(hashlib.md5(key).digest()[:4], "little") & 0x7FFFFFFF


# ----------------------------------------------------------------------------------------------
# Lecture du jeu
# ----------------------------------------------------------------------------------------------
def read_yolo_labels(label_path: str):
    bboxes, classes = [], []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                classes.append(int(float(parts[0])))
                bboxes.append([float(v) for v in parts[1:5]])
    return bboxes, classes


def write_yolo_labels(label_path: str, bboxes, classes):
    with open(label_path, "w") as f:
        for cls, (cx, cy, w, h) in zip(classes, bboxes):
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def load_items(cfg: AlignedAugConfig):
    names = sorted(n for n in os.listdir(cfg.images_dir) if n.lower().endswith(IMAGE_EXTENSIONS))
    items = []
    for n in names:
        stem = os.path.splitext(n)[0]
        bboxes, classes = read_yolo_labels(os.path.join(cfg.labels_dir, stem + ".txt"))
        items.append({"name": n, "stem": stem, "bboxes": bboxes, "classes": classes})
    return items


# ----------------------------------------------------------------------------------------------
# Allocation : la SEULE partie qui diffère entre naïf (alpha = 0) et class-aware (alpha > 0)
# ----------------------------------------------------------------------------------------------
def resolve_boost_classes(boost_classes, class_names) -> set | None:
    """Convertit boost_classes (ids ou noms) en ensemble d'ids. () -> None (toutes les classes)."""
    if not boost_classes:
        return None
    name_to_id = {v: k for k, v in (class_names or {}).items()}
    ids = set()
    for c in boost_classes:
        if isinstance(c, (int, np.integer)):
            ids.add(int(c))
        elif c in name_to_id:
            ids.add(int(name_to_id[c]))
        else:
            raise ValueError(f"boost_classes : classe inconnue {c!r} (vérifiez class_names).")
    return ids


def image_weights(items, alpha: float, boost_ids: set | None = None):
    """Poids d'une image = max des poids de rareté de ses classes.
    Poids de rareté d'une classe c : n_c ** (-alpha) si c est ciblée, sinon le poids de base
    (celui de la classe la plus fréquente, le plus faible). Sans boost_ids, toutes les classes
    sont ciblées. Une image sans classe ciblée (ou sans annotation) reçoit le poids de base."""
    n_img_per_class = Counter(c for it in items for c in set(it["classes"]))
    if alpha == 0 or not n_img_per_class:
        return np.ones(len(items), dtype=np.float64), n_img_per_class
    base = min(n ** (-alpha) for n in n_img_per_class.values())
    targeted = set(n_img_per_class) if boost_ids is None else boost_ids
    weights = []
    for it in items:
        present = set(it["classes"]) & targeted
        w = max([base] + [n_img_per_class[c] ** (-alpha) for c in present])
        weights.append(w)
    return np.asarray(weights, dtype=np.float64), n_img_per_class


def allocate(weights: np.ndarray, budget: int, k_max: int, rng: random.Random) -> np.ndarray:
    """Répartit `budget` copies proportionnellement aux poids, plafonnées à k_max (remplissage
    par niveaux), puis arrondit par plus grands restes : sum == budget exactement."""
    n = len(weights)
    if budget > k_max * n:
        raise ValueError(f"Budget {budget} impossible avec k_max={k_max} et {n} images (max {k_max * n}).")
    share = np.zeros(n)
    active = np.ones(n, dtype=bool)
    remaining = float(budget)
    while active.any():
        s = remaining * weights[active] / weights[active].sum()
        over = s > k_max + 1e-12
        if not over.any():
            share[active] = s
            break
        idx = np.flatnonzero(active)[over]
        share[idx] = k_max
        active[idx] = False
        remaining -= k_max * len(idx)

    k = np.floor(share + 1e-9).astype(int)
    missing = budget - int(k.sum())
    if missing > 0:
        frac = share - k
        order = sorted((i for i in range(n) if k[i] < k_max), key=lambda i: (-round(frac[i], 9), rng.random()))
        for i in order[:missing]:
            k[i] += 1
    assert int(k.sum()) == budget
    return k


def image_catalog(classes, swap_ids, swap_mode="transpose"):
    """Plan de transformations d'une image : dict avec
      catalog     : les 7 transformations des copies 1..7 ;
      cycle_extra : ajoutées à la fin de chaque cycle (cycle répété) ;
      once_extra  : ajoutées une seule fois après les 7 premières ;
      tail        : si non vide, après once_extra on répète UNIQUEMENT ces transformations ;
      swapped     : True si l'image est une image de carence traitée par swap_mode.
    Images contenant une classe de swap_ids (carences) :
      * "transpose"          : couleurs -> transpose / anti_transpose ; cycle des 7 répété.
      * "illumination_cycle" : couleurs -> shadow / sunflare ; cycle de 10 répété =
                               7 + transpose + anti_transpose + combinaison tcc.
      * "transpose_cycle"    : 7 du naïf (identiques) ; puis une fois transpose + rgb_shift ±10 et
                               anti_transpose + brightness_contrast ±0,1 ; puis les 5 géométriques en boucle.
    Autres images : catalogue du naïf, cycle des 7 répété."""
    plan = dict(catalog=TRANSFORM_NAMES, cycle_extra=(), once_extra=(), tail=(), swapped=False)
    if swap_ids and set(classes) & swap_ids:
        plan["swapped"] = True
        if swap_mode == "illumination_cycle":
            plan.update(catalog=tuple(COLOR_TO_ILLUMINATION.get(t, t) for t in TRANSFORM_NAMES),
                        cycle_extra=ILLUMINATION_CYCLE_EXTRA)
        elif swap_mode == "transpose_cycle":
            plan.update(once_extra=TRANSPOSE_CYCLE_ONCE, tail=GEOMETRIC)
        else:
            plan.update(catalog=tuple(COLOR_TO_TRANSPOSE.get(t, t) for t in TRANSFORM_NAMES))
    return plan


def transform_schedule(stem: str, k: int, seed: int, extra_mode: str = "repeat", catalog=TRANSFORM_NAMES,
                       cycle_extra=(), once_extra=(), tail=()):
    """Liste des (transformation, répétition) pour une image : on parcourt les 7 transformations du
    catalogue de l'image dans un ordre mélangé propre à l'image, puis, si k > 7 :
      * extra_mode="repeat"    : on recommence le cycle (défaut). Avec cycle_extra, le cycle est
                                 [7 du catalogue mélangés] + cycle_extra (ex. cycle de 10).
                                 Avec once_extra / tail : 7, puis once_extra une fois, puis tail en boucle ;
      * extra_mode="d4_affine" : les copies 8..k sont toutes "d4_affine" (répétitions 1..k-7).
    Avec k <= 7 (tout le naïf), le calendrier est identique dans tous les modes."""
    order = list(catalog)
    random.Random(sample_seed(seed, stem, "order", 0, 0)).shuffle(order)
    n = len(order)
    if extra_mode == "d4_affine":
        return [(order[r], 0) if r < n else (EXTRA_D4_AFFINE, r - n + 1) for r in range(k)]
    if tail:
        seq = [(t, 0) for t in order] + [(t, 0) for t in once_extra]
        tail_order = [t for t in order if t in tail]   # même ordre que les 7 premières
        j = 0
        while len(seq) < k:
            seq.append((tail_order[j % len(tail_order)], 1 + j // len(tail_order)))
            j += 1
        return seq[:k]
    cycle = order + list(cycle_extra)
    L = len(cycle)
    out = []
    for r in range(k):
        name, rep = cycle[r % L], r // L
        if name == "tcc":  # combinaison : transposition tirée au hasard pour ce cycle
            pick = random.Random(sample_seed(seed, stem, "tcc_pick", rep, 0)).random() < 0.5
            name = "tcc_transpose" if pick else "tcc_anti_transpose"
        out.append((name, rep))
    return out


def retry_candidates(stem: str, name: str, rep: int, seed: int, catalog=TRANSFORM_NAMES):
    """Ordre des essais pour une copie (utilisé seulement si dedupe ou retry_failed) :
    d'abord la transformation prévue, retentée avec de nouvelles graines (corrige par ex. le
    RandomRotate90 qui tire 0°), puis les 6 autres dans un ordre aléatoire propre à la copie
    (pas de biais systématique vers une transformation de repli)."""
    if name == EXTRA_D4_AFFINE:  # transformation continue : de nouvelles graines suffisent
        return [name] * (N_SAME_TRANSFORM_RETRIES + len(TRANSFORM_NAMES))
    others = [t for t in catalog if t != name]
    rng = random.Random(sample_seed(seed, stem, name, rep, -1))
    rng.shuffle(others)
    # derniers recours : les transformations continues du catalogue (toujours une copie nouvelle)
    continuous = [t for t in catalog if t in ("rotate", "random_crop")] * 5
    rng.shuffle(continuous)
    return [name] * N_SAME_TRANSFORM_RETRIES + others + continuous


# ----------------------------------------------------------------------------------------------
# Copy-paste léger (extra_mode="d4_affine_cp")
# ----------------------------------------------------------------------------------------------
class CopyPasteContext:
    """Index des lésions des classes ciblées (train uniquement) et collage non invasif."""

    def __init__(self, items, boost_ids, n_img_per_class, cfg):
        self.items, self.cfg = items, cfg
        self.n_img = n_img_per_class
        self.targets = set(boost_ids) if boost_ids is not None else set(n_img_per_class)
        self.pool = defaultdict(list)  # classe -> [(index image, boîte yolo)]
        for i, it in enumerate(items):
            for b, c in zip(it["bboxes"], it["classes"]):
                if c in self.targets:
                    self.pool[c].append((i, b))
        self._cache = {}

    def _read(self, i):
        if i not in self._cache:
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
            im = cv2.imread(os.path.join(self.cfg.images_dir, self.items[i]["name"]))
            self._cache[i] = None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return self._cache[i]

    def target_class(self, it):
        """Classe ciblée la plus rare présente dans l'image (None si aucune)."""
        present = [c for c in set(it["classes"]) if c in self.targets and self.pool[c]]
        return min(present, key=lambda c: self.n_img[c]) if present else None

    def paste(self, img, bboxes, classes, src_index, cls, rng):
        """Colle au plus cp_max_patches lésions de `cls`. Retourne (img, bboxes, classes, n_collées)."""
        cfg = self.cfg
        H, W = img.shape[:2]
        out = img.copy()
        boxes_px = [_yolo_to_px(b, W, H) for b in bboxes]
        margin = cfg.cp_margin * max(H, W)
        candidates = [e for e in self.pool[cls] if e[0] != src_index]
        n_done = 0
        for _ in range(cfg.cp_max_patches):
            if not candidates:
                break
            j, box = rng.choice(candidates)
            src = self._read(j)
            if src is None:
                continue
            sh, sw = src.shape[:2]
            x1, y1, x2, y2 = (int(round(v)) for v in _yolo_to_px(box, sw, sh))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(sw, x2), min(sh, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            patch = src[y1:y2, x1:x2]
            # même taille RELATIVE que dans l'image d'origine, plafonnée
            s = math.hypot(W, H) / math.hypot(sw, sh)
            s = min(s, cfg.cp_max_rel_size * W / patch.shape[1], cfg.cp_max_rel_size * H / patch.shape[0])
            pw, ph = int(round(patch.shape[1] * s)), int(round(patch.shape[0] * s))
            if pw < 6 or ph < 6 or pw >= W or ph >= H:
                continue
            patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            for _try in range(cfg.cp_position_tries):
                px, py = rng.randint(0, W - pw), rng.randint(0, H - ph)
                cand = (px, py, px + pw, py + ph)
                if all(not _overlap(cand, b, margin) for b in boxes_px):
                    break
            else:
                continue  # aucune place libre : on ne colle pas (non invasif)
            alpha = _feather_mask(ph, pw, cfg.cp_feather)[..., None]
            region = out[py:py + ph, px:px + pw].astype(np.float32)
            out[py:py + ph, px:px + pw] = (alpha * patch + (1 - alpha) * region).round().astype(np.uint8)
            boxes_px.append(cand)
            bboxes = list(bboxes) + [_px_to_yolo(cand, W, H)]
            classes = list(classes) + [int(cls)]
            n_done += 1
        return out, bboxes, classes, n_done


def _yolo_to_px(b, W, H):
    cx, cy, w, h = b
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def _px_to_yolo(b, W, H):
    x1, y1, x2, y2 = b
    return [min(max((x1 + x2) / 2 / W, 0.0), 1.0), min(max((y1 + y2) / 2 / H, 0.0), 1.0),
            min((x2 - x1) / W, 1.0), min((y2 - y1) / H, 1.0)]


def _overlap(a, b, margin):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def _feather_mask(h, w, feather):
    """Masque alpha : 1 au centre, descend progressivement à 0 sur une bordure de `feather` × taille."""
    if feather <= 0:
        return np.ones((h, w), np.float32)
    fy, fx = max(1.0, feather * h), max(1.0, feather * w)
    ys = np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5) / fy
    xs = np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5) / fx
    return np.clip(np.minimum.outer(ys, xs), 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Génération
# ----------------------------------------------------------------------------------------------
def _apply(img, it, name, rep, attempt, cfg, cp_ctx=None, src_index=None):
    """Retourne (image, boîtes, classes, secondaire ou None, nb de lésions collées)."""
    h, w = img.shape[:2]
    bboxes, classes, n_pasted = it["bboxes"], it["classes"], 0
    if cp_ctx is not None and name == EXTRA_D4_AFFINE:
        # décision de coller : indépendante de l'essai (sinon les nouveaux essais biaisent cp_prob)
        decide = random.Random(sample_seed(cfg.seed, it["stem"], "copy_paste_decision", rep, 0)).random()
        rng = random.Random(sample_seed(cfg.seed, it["stem"], "copy_paste", rep, attempt))
        cls = cp_ctx.target_class(it)
        if cls is not None and decide < cfg.cp_prob:
            img, bboxes, classes, n_pasted = cp_ctx.paste(img, bboxes, classes, src_index, cls, rng)
    second = secondary_transform(it["stem"], name, rep, attempt, cfg)
    tf = build_transform((name, second) if second else name, h, w, cfg)
    tf.set_random_seed(sample_seed(cfg.seed, it["stem"], name, rep, attempt))
    out = tf(image=img, bboxes=bboxes, class_labels=classes)
    return out["image"], list(out["bboxes"]), [int(c) for c in out["class_labels"]], second, n_pasted


def generate(cfg: AlignedAugConfig):
    rng = random.Random(cfg.seed)
    os.makedirs(cfg.out_images_dir, exist_ok=True)
    os.makedirs(cfg.out_labels_dir, exist_ok=True)

    if cfg.extra_mode not in ("repeat", "d4_affine", "d4_affine_cp"):
        raise ValueError(f"extra_mode inconnu : {cfg.extra_mode!r} (attendu 'repeat', 'd4_affine' ou 'd4_affine_cp').")
    if cfg.extra_mode != "repeat" and cfg.combine:
        raise ValueError("extra_mode='d4_affine…' et combine=True sont deux ablations distinctes : choisir l'une.")
    schedule_mode = "d4_affine" if cfg.extra_mode.startswith("d4_affine") else "repeat"

    items = load_items(cfg)
    n_src = len(items)
    budget = cfg.budget if cfg.budget is not None else len(TRANSFORM_NAMES) * n_src
    k_max = cfg.k_max if cfg.alpha > 0 else max(cfg.k_max, math.ceil(budget / n_src))

    boost_ids = resolve_boost_classes(cfg.boost_classes, cfg.class_names)
    swap_ids = resolve_boost_classes(cfg.swap_classes, cfg.class_names)
    if cfg.swap_mode not in ("transpose", "illumination_cycle", "transpose_cycle"):
        raise ValueError(f"swap_mode inconnu : {cfg.swap_mode!r} "
                         "(attendu 'transpose', 'illumination_cycle' ou 'transpose_cycle').")
    if swap_ids and (cfg.combine or cfg.extra_mode != "repeat"):
        raise ValueError("swap_classes s'utilise avec extra_mode='repeat' et combine=False.")
    weights, n_img_per_class = image_weights(items, cfg.alpha, boost_ids)
    k = allocate(weights, budget, k_max, rng)
    cp_ctx = (CopyPasteContext(items, boost_ids, n_img_per_class, cfg)
              if cfg.extra_mode == "d4_affine_cp" else None)
    pasted_counts = Counter()
    n_swapped_images = 0

    before = Counter(c for it in items for c in it["classes"])
    after = Counter(before) if cfg.copy_originals else Counter()
    transform_counts = Counter()
    pair_counts = Counter()
    manifest = []
    n_dup_written, n_rejected = 0, 0

    for src_index, (it, k_i) in enumerate(zip(items, k)):
        src_path = os.path.join(cfg.images_dir, it["name"])
        if cfg.copy_originals:
            shutil.copy(src_path, os.path.join(cfg.out_images_dir, it["name"]))
            lp = os.path.join(cfg.labels_dir, it["stem"] + ".txt")
            if os.path.exists(lp):
                shutil.copy(lp, os.path.join(cfg.out_labels_dir, it["stem"] + ".txt"))
        if k_i == 0:
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f"[ATTENTION] illisible : {src_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seen = {hashlib.md5(img.tobytes()).hexdigest()}
        had_boxes = len(it["bboxes"]) > 0

        plan = image_catalog(it["classes"], swap_ids, cfg.swap_mode)
        catalog = plan["catalog"]
        n_swapped_images += plan["swapped"]
        for name, rep in transform_schedule(it["stem"], int(k_i), cfg.seed, schedule_mode, catalog,
                                            plan["cycle_extra"], plan["once_extra"], plan["tail"]):
            # copies de la boucle finale (tail) : replis pris uniquement dans la tail (pas de couleur)
            retry_cat = plan["tail"] if (plan["tail"] and rep >= 1) else catalog
            candidates = retry_candidates(it["stem"], name, rep, cfg.seed, retry_cat)
            written, logged, last_status = False, False, "exhausted"
            for attempt, cand in enumerate(candidates):
                if attempt > 0 and not (cfg.dedupe or cfg.retry_failed):
                    break
                status, second, n_pasted = "ok", None, 0
                try:
                    aug, bxs, cls, second, n_pasted = _apply(img, it, cand, rep, attempt, cfg, cp_ctx, src_index)
                except Exception as e:  # noqa: BLE001 — même comportement que le script naïf
                    status, aug = f"error:{type(e).__name__}", None
                if aug is not None and had_boxes and len(bxs) == 0:
                    status = "boxes_lost"
                if status != "ok":
                    if cfg.retry_failed:
                        last_status = status
                        continue
                    n_rejected += 1
                    manifest.append({"file": "", "source": it["name"], "transform": cand, "rep": rep,
                                     "attempt": attempt, "status": status, "n_boxes": 0})
                    logged = True
                    break
                h_out = hashlib.md5(aug.tobytes()).hexdigest()
                is_dup = h_out in seen
                if is_dup and cfg.dedupe:
                    last_status = "duplicate_exhausted"
                    continue
                seen.add(h_out)
                n_dup_written += int(is_dup)

                combo = f"{cand}+{second}" if second else cand
                if n_pasted:
                    combo += "+cp"
                    pasted_counts[cp_ctx.target_class(it)] += n_pasted
                out_stem = f"{it['stem']}_{combo}_r{rep}" + (f"_a{attempt}" if attempt else "")
                cv2.imwrite(os.path.join(cfg.out_images_dir, out_stem + ".jpg"),
                            cv2.cvtColor(aug, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
                write_yolo_labels(os.path.join(cfg.out_labels_dir, out_stem + ".txt"), bxs, cls)
                after.update(cls)
                transform_counts[cand] += 1
                if second:
                    pair_counts[combo] += 1
                manifest.append({"file": out_stem + ".jpg", "source": it["name"], "transform": combo,
                                 "rep": rep, "attempt": attempt,
                                 "status": "duplicate" if is_dup else "ok", "n_boxes": len(bxs)})
                written = True
                break
            if not written and not logged:
                n_rejected += 1
                manifest.append({"file": "", "source": it["name"], "transform": name, "rep": rep,
                                 "attempt": -1, "status": last_status, "n_boxes": 0})

    summary = _summarize(cfg, items, k, budget, k_max, n_img_per_class,
                         before, after, transform_counts, manifest, n_dup_written, n_rejected)
    summary["pair_counts"] = dict(sorted(pair_counts.items()))
    names_ = cfg.class_names or {}
    summary["pasted_lesions"] = {names_.get(c, str(c)): n for c, n in sorted(pasted_counts.items())}
    summary["images_with_swapped_catalog"] = n_swapped_images
    _write_reports(cfg, manifest, summary)
    return summary


# ----------------------------------------------------------------------------------------------
# Rapports
# ----------------------------------------------------------------------------------------------
def _summarize(cfg, items, k, budget, k_max, n_img_per_class, before, after,
               transform_counts, manifest, n_dup_written, n_rejected):
    names = cfg.class_names or {}
    label = lambda c: names.get(c, str(c))  # noqa: E731
    written = sum(1 for m in manifest if m["file"])
    copies_per_class = defaultdict(list)
    for it, k_i in zip(items, k):
        for c in set(it["classes"]):
            copies_per_class[c].append(int(k_i))
    return {
        "method": "naive" if cfg.alpha == 0 else "class_aware",
        "config": {kk: vv for kk, vv in asdict(cfg).items() if kk != "class_names"},
        "n_source_images": len(items),
        "budget_planned": int(budget),
        "copies_written": written,
        "copies_rejected": n_rejected,
        "exact_duplicates_written": n_dup_written,
        "k_max_effective": int(k_max),
        "copies_per_source": {"min": int(k.min()), "mean": float(k.mean()), "max": int(k.max()),
                              "n_sources_with_0": int((k == 0).sum())},
        "copies_per_source_by_class_mean": {label(c): float(np.mean(v)) for c, v in sorted(copies_per_class.items())},
        "source_images_per_class": {label(c): n for c, n in sorted(n_img_per_class.items())},
        "instances_before": {label(c): n for c, n in sorted(before.items())},
        "instances_after": {label(c): n for c, n in sorted(after.items())},
        "transform_counts": dict(transform_counts),
    }


def _write_reports(cfg, manifest, summary):
    report_dir = os.path.dirname(os.path.normpath(cfg.out_images_dir))
    with open(os.path.join(report_dir, "augmentation_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "source", "transform", "rep", "attempt", "status", "n_boxes"])
        w.writeheader()
        w.writerows(manifest)
    with open(os.path.join(report_dir, "augmentation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_summary(s):
    boost = s["config"]["boost_classes"] or "toutes"
    print(f"Méthode : {s['method']}  (alpha={s['config']['alpha']}, k_max={s['k_max_effective']}, "
          f"classes ciblées : {boost})")
    print(f"Images sources : {s['n_source_images']} | budget : {s['budget_planned']} | "
          f"écrites : {s['copies_written']} | rejetées : {s['copies_rejected']} | "
          f"doublons exacts écrits : {s['exact_duplicates_written']}")
    print(f"Copies par source : {s['copies_per_source']}")
    print("Copies moyennes par source, selon la classe présente :", s["copies_per_source_by_class_mean"])
    print("Instances avant :", s["instances_before"])
    print("Instances après :", s["instances_after"])
    print("Transformations :", s["transform_counts"])
    if s.get("pair_counts"):
        print("Combinaisons (principale+secondaire) :", s["pair_counts"])
    if s.get("images_with_swapped_catalog"):
        print(f"Images de carence au catalogue modifié ({s['config']['swap_mode']}) :",
              s["images_with_swapped_catalog"])
    if s.get("pasted_lesions"):
        print("Lésions collées (copy-paste) :", s["pasted_lesions"])
