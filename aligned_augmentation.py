
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

TRANSFORM_NAMES2 = (
    "rgb_shift",
    "rgb_shift",
    "rotate",
    "random_rotate90",
    "horizontal_flip",
    "vertical_flip",
    "random_crop",
)

GEOMETRIC = ("rotate", "random_rotate90", "horizontal_flip", "vertical_flip", "random_crop")
CONTINUOUS = ("rotate", "random_crop")                      # toujours une copie nouvelle (derniers recours)
TRANSPOSE_CYCLE_ONCE = ("transpose_rgb_shift", "anti_transpose_bc")
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
    budget: int | None = None             # None -> 7 × nombre d'images (budget du naïf)
    k_max: int = 28                       # plafond de copies par image (CA)
    boost_classes: tuple = ()             # classes ciblées par la CA (ids ou noms) ; () = toutes
    class_names: dict = field(default_factory=dict)  # {id: nom}

    swap_mode: str | None = None          # None ou "transpose_cycle"
    swap_classes: tuple = ()              # classes « carence » pour swap_mode, ex. (4, 5, 6) = Mg, N, K
    tc_rgb_shift: int = 5                 # copie 8 : RGBShift ±5 avec la transposition
    tac_bc_limit: float = 0.5             # copie 9 : luminosité/contraste ±0,5 avec l'anti-transposition

    copy_originals: bool = True
    dedupe: bool = False
    retry_failed: bool = False
    min_visibility: float = 0.2
    crop_fraction: float = 0.7
    seed: int = 42
    jpeg_quality: int = 95


# ----------------------------------------------------------------------------------------------
# Transformations
# ----------------------------------------------------------------------------------------------
def _single_transform(name: str, img_h: int, img_w: int, cfg: AlignedAugConfig):
    """Une transformation, ou une liste de transformations appliquées dans l'ordre."""
    if name == "rgb_shift":
        return A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)
    if name == "brightness_contrast":
        return A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
    if name == "rotate":
        return A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=1.0)
    if name == "random_rotate90":
        return A.RandomRotate90(p=1.0)
    if name == "horizontal_flip":
        return A.HorizontalFlip(p=1.0)
    if name == "vertical_flip":
        return A.VerticalFlip(p=1.0)
    if name == "random_crop":
        return A.RandomCrop(height=int(img_h * cfg.crop_fraction), width=int(img_w * cfg.crop_fraction), p=1.0)
    if name == "transpose_rgb_shift":     # couleur légère puis symétrie sur la diagonale
        s = cfg.tc_rgb_shift
        return [A.RGBShift(r_shift_limit=s, g_shift_limit=s, b_shift_limit=s, p=1.0), A.Transpose(p=1.0)]
    if name == "anti_transpose_bc":       # luminosité/contraste puis symétrie sur l'anti-diagonale
        b = cfg.tac_bc_limit              # (anti-transposition = transposition puis rotation de 180°)
        return [A.RandomBrightnessContrast(brightness_limit=b, contrast_limit=b, p=1.0),
                A.Transpose(p=1.0), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]
    raise ValueError(f"Transformation inconnue : {name}")


def build_transform(name: str, img_h: int, img_w: int, cfg: AlignedAugConfig) -> A.Compose:
    t = _single_transform(name, img_h, img_w, cfg)
    # liste à plat dans Compose : un Sequential imbriqué ne reçoit pas la graine de Compose
    return A.Compose(t if isinstance(t, list) else [t],
                     bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                              min_visibility=cfg.min_visibility))


def sample_seed(seed: int, stem: str, name: str, rep: int, attempt: int) -> int:
    """Graine déterministe par copie : même (image, transformation, répétition, essai) -> même copie."""
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
                if parts:
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


def resolve_classes(classes, class_names) -> set | None:
    """Ids ou noms -> ensemble d'ids. () -> None."""
    if not classes:
        return None
    name_to_id = {v: k for k, v in (class_names or {}).items()}
    ids = set()
    for c in classes:
        if isinstance(c, (int, np.integer)):
            ids.add(int(c))
        elif c in name_to_id:
            ids.add(int(name_to_id[c]))
        else:
            raise ValueError(f"Classe inconnue : {c!r} (vérifiez class_names).")
    return ids


# ----------------------------------------------------------------------------------------------
# Allocation : la seule différence entre naïf (alpha = 0) et CA (alpha > 0)
# ----------------------------------------------------------------------------------------------
def image_weights(items, alpha: float, boost_ids: set | None = None):
    n_img_per_class = Counter(c for it in items for c in set(it["classes"]))
    if alpha == 0 or not n_img_per_class:
        return np.ones(len(items), dtype=np.float64), n_img_per_class
    base = min(n ** (-alpha) for n in n_img_per_class.values())
    targeted = set(n_img_per_class) if boost_ids is None else boost_ids
    weights = [max([base] + [n_img_per_class[c] ** (-alpha) for c in set(it["classes"]) & targeted])
               for it in items]
    return np.asarray(weights, dtype=np.float64), n_img_per_class


def allocate(weights: np.ndarray, budget: int, k_max: int, rng: random.Random) -> np.ndarray:
    """Copies proportionnelles aux poids, plafonnées à k_max, somme = budget exactement."""
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


# ----------------------------------------------------------------------------------------------
# Calendrier des copies
# ----------------------------------------------------------------------------------------------
def transform_schedule(stem: str, k: int, seed: int, swapped: bool = False):
    """Liste des (transformation, répétition) d'une image.
    Normal  : les 7 dans un ordre mélangé propre à l'image, puis le cycle recommence.
    swapped : les 7, puis transpose_rgb_shift, anti_transpose_bc, puis les 5 géométriques en boucle
              (dans le même ordre que pour les 7 premières)."""
    order = list(TRANSFORM_NAMES)
    order2 = list(TRANSFORM_NAMES)
    random.Random(sample_seed(seed, stem, "order", 0, 0)).shuffle(order)
    n = len(order)
    if not swapped:
        return [(order2[r % n], r // n) for r in range(k)]
    seq = [(t, 0) for t in order] + [(t, 0) for t in TRANSPOSE_CYCLE_ONCE]
    #tail = [t for t in order if t in GEOMETRIC]
    tail = [t for t in order if t in TRANSFORM_NAMES]
    j = 0
    while len(seq) < k:
        seq.append((tail[j % len(tail)], 1 + j // len(tail)))
        j += 1
    return seq[:k]


def retry_candidates(stem: str, name: str, rep: int, seed: int, catalog=TRANSFORM_NAMES):
    """Essais pour une copie (si dedupe ou retry_failed) : la transformation prévue avec de nouvelles graines,
    puis les autres du catalogue (ordre aléatoire propre à la copie), puis rotate / random_crop en dernier
    recours (elles donnent toujours une copie nouvelle)."""
    others = [t for t in catalog if t != name]
    rng = random.Random(sample_seed(seed, stem, name, rep, -1))
    rng.shuffle(others)
    last = [t for t in catalog if t in CONTINUOUS] * 5
    rng.shuffle(last)
    return [name] * N_SAME_TRANSFORM_RETRIES + others + last


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
    if cfg.swap_mode not in (None, "transpose_cycle"):
        raise ValueError(f"swap_mode inconnu : {cfg.swap_mode!r} (attendu None ou 'transpose_cycle').")
    swap_ids = resolve_classes(cfg.swap_classes, cfg.class_names) if cfg.swap_mode else None
    if cfg.swap_mode and not swap_ids:
        raise ValueError("swap_mode='transpose_cycle' demande swap_classes, ex. (4, 5, 6).")

    rng = random.Random(cfg.seed)
    os.makedirs(cfg.out_images_dir, exist_ok=True)
    os.makedirs(cfg.out_labels_dir, exist_ok=True)

    items = load_items(cfg)
    n_src = len(items)
    budget = cfg.budget if cfg.budget is not None else len(TRANSFORM_NAMES) * n_src
    k_max = cfg.k_max if cfg.alpha > 0 else max(cfg.k_max, math.ceil(budget / n_src))
    boost_ids = resolve_classes(cfg.boost_classes, cfg.class_names)
    weights, n_img_per_class = image_weights(items, cfg.alpha, boost_ids)
    k = allocate(weights, budget, k_max, rng)

    before = Counter(c for it in items for c in it["classes"])
    after = Counter(before) if cfg.copy_originals else Counter()
    transform_counts = Counter()
    manifest = []
    n_dup_written = n_rejected = n_swapped = 0

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
        swapped = bool(swap_ids and set(it["classes"]) & swap_ids)
        n_swapped += swapped

        for name, rep in transform_schedule(it["stem"], int(k_i), cfg.seed, swapped):
            # copies 10+ d'une image de carence : replis pris parmi les 5 géométriques (pas de couleur)
            catalog = GEOMETRIC if (swapped and rep >= 1) else TRANSFORM_NAMES
            written, logged, last_status = False, False, "exhausted"
            for attempt, cand in enumerate(retry_candidates(it["stem"], name, rep, cfg.seed, catalog)):
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
                            cv2.cvtColor(aug, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
                write_yolo_labels(os.path.join(cfg.out_labels_dir, out_stem + ".txt"), bxs, cls)
                after.update(cls)
                transform_counts[cand] += 1
                manifest.append({"file": out_stem + ".jpg", "source": it["name"], "transform": cand, "rep": rep,
                                 "attempt": attempt, "status": "duplicate" if is_dup else "ok",
                                 "n_boxes": len(bxs)})
                written = True
                break
            if not written and not logged:
                n_rejected += 1
                manifest.append({"file": "", "source": it["name"], "transform": name, "rep": rep,
                                 "attempt": -1, "status": last_status, "n_boxes": 0})

    summary = _summarize(cfg, items, k, budget, k_max, n_img_per_class, before, after,
                         transform_counts, manifest, n_dup_written, n_rejected, n_swapped)
    _write_reports(cfg, manifest, summary)
    return summary


# ----------------------------------------------------------------------------------------------
# Rapports
# ----------------------------------------------------------------------------------------------
def _summarize(cfg, items, k, budget, k_max, n_img_per_class, before, after,
               transform_counts, manifest, n_dup_written, n_rejected, n_swapped):
    names = cfg.class_names or {}
    label = lambda c: names.get(c, str(c))  # noqa: E731
    copies_per_class = defaultdict(list)
    for it, k_i in zip(items, k):
        for c in set(it["classes"]):
            copies_per_class[c].append(int(k_i))
    return {
        "method": ("naive" if cfg.alpha == 0 else "class_aware") + ("+transpose_cycle" if cfg.swap_mode else ""),
        "config": {kk: vv for kk, vv in asdict(cfg).items() if kk != "class_names"},
        "n_source_images": len(items),
        "budget_planned": int(budget),
        "copies_written": sum(1 for m in manifest if m["file"]),
        "copies_rejected": n_rejected,
        "exact_duplicates_written": n_dup_written,
        "k_max_effective": int(k_max),
        "images_with_transpose_cycle": n_swapped,
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
    c = s["config"]
    print(f"Méthode : {s['method']}  (alpha={c['alpha']}, k_max={s['k_max_effective']}, "
          f"classes ciblées : {c['boost_classes'] or 'toutes'})")
    print(f"Images sources : {s['n_source_images']} | budget : {s['budget_planned']} | "
          f"écrites : {s['copies_written']} | rejetées : {s['copies_rejected']} | "
          f"doublons exacts écrits : {s['exact_duplicates_written']}")
    if c["swap_mode"]:
        print(f"Images de carence en transpose_cycle : {s['images_with_transpose_cycle']} "
              f"(RGBShift ±{c['tc_rgb_shift']}, luminosité/contraste ±{c['tac_bc_limit']})")
    print(f"Copies par source : {s['copies_per_source']}")
    print("Copies moyennes par source, selon la classe présente :", s["copies_per_source_by_class_mean"])
    print("Instances avant :", s["instances_before"])
    print("Instances après :", s["instances_after"])
    print("Transformations :", s["transform_counts"])