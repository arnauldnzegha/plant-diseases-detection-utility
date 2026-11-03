"""
Augmentation hors-ligne ALIGNÉE : DA naïve et DA class-aware avec EXACTEMENT les mêmes transformations.

Principe (pour une comparaison équitable dans le papier)
--------------------------------------------------------
Les deux méthodes partagent :
  * le même catalogue de 7 transformations, avec les mêmes paramètres (ceux du papier source),
  * la même gestion des boîtes (BboxParams identiques),
  * le même tirage aléatoire par échantillon (graine dérivée de (seed, image, transformation, répétition)),
  * le même budget total de copies augmentées,
  * les mêmes options (dedupe, retry_failed).

Seule la POLITIQUE D'ALLOCATION change, contrôlée par un unique paramètre `alpha` :
  * alpha = 0  -> chaque image reçoit exactement 7 copies (une par transformation) = DA naïve du papier source.
  * alpha > 0  -> les copies sont redistribuées vers les images contenant des classes rares (class-aware),
                  le budget total restant identique.

Poids d'une image i :  w_i = max_{c présente dans i} (n_c) ** (-alpha)
  où n_c = nombre d'images de train contenant la classe c (l'image "vaut" sa classe la plus rare).
Avec `boost_classes`, seules les classes listées reçoivent ce poids de rareté ; les autres (et les
images sans annotation) reçoivent le poids de la classe la plus fréquente. Sans cette option, quand les
classes les plus rares atteignent k_max, le budget restant se reporte sur la classe rare suivante
(par ex. Mg), ce qui peut renforcer un biais au lieu de le corriger.
Les copies k_i sont proportionnelles à w_i, plafonnées à k_max, puis arrondies (plus grands restes)
de sorte que sum(k_i) == budget exactement.

Les 7 transformations (une seule par copie, p = 1)
--------------------------------------------------
  rgb_shift            RGBShift(±20, ±20, ±20)
  brightness_contrast  RandomBrightnessContrast(±0.2, ±0.2)
  rotate               Rotate(limit=90, bord constant)        (angle aléatoire dans [-90°, 90°])
  random_rotate90      RandomRotate90                          (peut tirer 0° -> copie identique, cf. `dedupe`)
  horizontal_flip      HorizontalFlip
  vertical_flip        VerticalFlip
  random_crop          RandomCrop(70 % H, 70 % W)               (relatif à la taille de chaque image)

Boîtes : format YOLO, min_visibility = 0.2. Une copie qui perd toutes ses boîtes (alors que l'image
source en avait) est rejetée, comme dans le script naïf d'origine.

Options partagées par les deux bras (à fixer à la MÊME valeur pour naïf et CA)
------------------------------------------------------------------------------
  dedupe       : une copie pixel-identique à l'original ou à une copie déjà produite de la même source
                 est retentée (même transformation avec une autre graine, puis une autre transformation).
                 False = fidèle au papier source.
  retry_failed : une copie rejetée (boîtes perdues / erreur) est retentée de la même façon, pour atteindre
                 exactement le budget. False = fidèle au papier source.

Avertissement : si une image reçoit plus de 7 copies, les flips reproduisent la même image.
Avec dedupe=False, ces doublons sont écrits ET comptés dans le rapport ; avec dedupe=True ils sont remplacés.

Testé avec albumentations 2.0.8 (Compose.set_random_seed). Vérifiez votre version.
"""

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
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
N_SAME_TRANSFORM_RETRIES = 3


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


def build_transform(name: str, img_h: int, img_w: int, cfg: AlignedAugConfig) -> A.Compose:
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
    else:
        raise ValueError(f"Transformation inconnue : {name}")
    return A.Compose([t], bbox_params=_bbox_params(cfg.min_visibility))


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


def transform_schedule(stem: str, k: int, seed: int):
    """Liste des (transformation, répétition) pour une image : on parcourt les 7 transformations
    dans un ordre mélangé propre à l'image, puis on recommence si k > 7.
    Avec k = 7 (naïf), chaque image reçoit chaque transformation exactement une fois."""
    order = list(TRANSFORM_NAMES)
    random.Random(sample_seed(seed, stem, "order", 0, 0)).shuffle(order)
    return [(order[r % len(order)], r // len(order)) for r in range(k)]


def retry_candidates(stem: str, name: str, rep: int, seed: int):
    """Ordre des essais pour une copie (utilisé seulement si dedupe ou retry_failed) :
    d'abord la transformation prévue, retentée avec de nouvelles graines (corrige par ex. le
    RandomRotate90 qui tire 0°), puis les 6 autres dans un ordre aléatoire propre à la copie
    (pas de biais systématique vers une transformation de repli)."""
    others = [t for t in TRANSFORM_NAMES if t != name]
    random.Random(sample_seed(seed, stem, name, rep, -1)).shuffle(others)
    return [name] * N_SAME_TRANSFORM_RETRIES + others


# ----------------------------------------------------------------------------------------------
# Génération
# ----------------------------------------------------------------------------------------------
def _apply(img, it, name, rep, attempt, cfg):
    h, w = img.shape[:2]
    tf = build_transform(name, h, w, cfg)
    tf.set_random_seed(sample_seed(cfg.seed, it["stem"], name, rep, attempt))
    out = tf(image=img, bboxes=it["bboxes"], class_labels=it["classes"])
    return out["image"], list(out["bboxes"]), [int(c) for c in out["class_labels"]]


def generate(cfg: AlignedAugConfig):
    rng = random.Random(cfg.seed)
    os.makedirs(cfg.out_images_dir, exist_ok=True)
    os.makedirs(cfg.out_labels_dir, exist_ok=True)

    items = load_items(cfg)
    n_src = len(items)
    budget = cfg.budget if cfg.budget is not None else len(TRANSFORM_NAMES) * n_src
    k_max = cfg.k_max if cfg.alpha > 0 else max(cfg.k_max, math.ceil(budget / n_src))

    boost_ids = resolve_boost_classes(cfg.boost_classes, cfg.class_names)
    weights, n_img_per_class = image_weights(items, cfg.alpha, boost_ids)
    k = allocate(weights, budget, k_max, rng)

    before = Counter(c for it in items for c in it["classes"])
    after = Counter(before) if cfg.copy_originals else Counter()
    transform_counts = Counter()
    manifest = []
    n_dup_written, n_rejected = 0, 0

    for it, k_i in zip(items, k):
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

        for name, rep in transform_schedule(it["stem"], int(k_i), cfg.seed):
            candidates = retry_candidates(it["stem"], name, rep, cfg.seed)
            written, logged, last_status = False, False, "exhausted"
            for attempt, cand in enumerate(candidates):
                if attempt > 0 and not (cfg.dedupe or cfg.retry_failed):
                    break
                status = "ok"
                try:
                    aug, bxs, cls = _apply(img, it, cand, rep, attempt, cfg)
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

                out_stem = f"{it['stem']}_{cand}_r{rep}" + (f"_a{attempt}" if attempt else "")
                cv2.imwrite(os.path.join(cfg.out_images_dir, out_stem + ".jpg"),
                            cv2.cvtColor(aug, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
                write_yolo_labels(os.path.join(cfg.out_labels_dir, out_stem + ".txt"), bxs, cls)
                after.update(cls)
                transform_counts[cand] += 1
                manifest.append({"file": out_stem + ".jpg", "source": it["name"], "transform": cand,
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
