import sys
sys.path.append('/content/drive/MyDrive/YOLO_KD/codes/')

"""
Génération HORS-LIGNE pour le rééquilibrage de TomatoVillage (format YOLO).

Ce que fait ce script (≠ dataloader en ligne) :
  1. lit train/images + train/labels,
  2. compte les instances par classe (AVANT),
  3. pour chaque classe sous-représentée, FABRIQUE des images augmentées (géométrie + photométrie
     + Copy-Paste de lésions rares) jusqu'à atteindre une cible,
  4. ÉCRIT sur disque les nouvelles images + leurs .txt dans un dossier de sortie,
  5. produit un rapport de distribution AVANT / APRÈS (texte, CSV, et graphique PNG).

-> Le dossier de sortie est un jeu d'entraînement ENRICHI, prêt à être pointé par data.yaml,
   puis (optionnellement) ré-augmenté en ligne par Ultralytics.

PRUDENCE conservée : letterbox, borne basse d'échelle, filtre des petites boîtes (< min_box_size_px).
On N'AUGMENTE QUE l'entraînement — ne lancez jamais ce script sur val/ ou test/.

Dépendances : albumentations, opencv-python, numpy, (matplotlib pour le graphique).
Place ce fichier à côté de tomato_dataloader.py (il en réutilise les fonctions).
"""

import os
import glob
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from tomato_dataloader import (
    AugConfig,
    yolo_to_xyxy,
    xyxy_to_yolo,
    filter_small_boxes,
    clip_boxes,
    build_geometric_photometric,
)


@dataclass
class OfflineConfig:
    images_dir: str = "/content/Tomato-Village/Variant-c(Object Detection)/train/images"
    labels_dir: str = "/content/Tomato-Village/Variant-c(Object Detection)/train/labels"
    out_images_dir: str = "/content/Tomato-Village/Variant-c(Object Detection)/train_balanced/images"
    out_labels_dir: str = "/content/Tomato-Village/Variant-c(Object Detection)/train_balanced/labels"
    copy_originals: bool = True       # recopie d'abord le jeu d'origine dans la sortie.

    # --- correspondance id YOLO -> nom (À ADAPTER selon votre data.yaml, champ `names:`) ---
    class_names: dict = field(default_factory=lambda: {
        0: "Early_blight",
        1: "Healty",
        2: "Late_blight",
        3: "Leaf Miner",
        4: "Magnesium Deficiency",
        5: "Nitrogen Deficiency",
        6: "Pottassium Deficiency",
        7: "Spotted Wilt Virus",
    })

    # --- CIBLES EXPLICITES par classe (clés = nom OU id). Classes absentes -> aucune génération. ---
    #   Plus lisible qu'un ratio global quand le déséquilibre est très inégal (cf. 869 vs 85 209).
    targets_per_class: dict = field(default_factory=lambda: {
        "Pottassium Deficiency": 8000,   # 869   -> ~8000  (×9,2)
        "Nitrogen Deficiency": 8000,     # 1 310 -> ~8000  (×6,1)
        "Late_blight": 12000,             # 3 174 -> ~8000  (×6)
    })

    # Repli : pour les classes SANS cible explicite, cible = max_count / target_ratio si > 0.
    target_ratio: float = 0.0             # 0 = désactivé (on n'utilise que targets_per_class).

    max_generated_per_class: int = 10000  # plafond de sécurité EN IMAGES (relevé : cibles élevées).
    beta_effective: float = 0.9999        # paramètre des poids "effective number of samples".
    seed: int = 0


def load_dataset(images_dir, labels_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = sorted(p for e in exts for p in glob.glob(os.path.join(images_dir, e)))
    items = []  # (img_path, label_path, boxes_n (N,4), labels (N,))
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(labels_dir, stem + ".txt")
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            data = np.loadtxt(lp, ndmin=2).astype(np.float32)
            labels = data[:, 0].astype(np.int64)
            boxes_n = data[:, 1:5]
        else:
            labels = np.zeros((0,), np.int64)
            boxes_n = np.zeros((0, 4), np.float32)
        items.append((p, lp, boxes_n, labels))
    return items


def count_instances(items):
    c = Counter()
    for _, _, _, labels in items:
        c.update(labels.tolist())
    return c


def resolve_targets(ocfg, before):
    """Construit {id_classe: instances_cible} à partir des cibles explicites (clé nom OU id),
    avec repli éventuel sur target_ratio pour les classes non listées."""
    name_to_id = {v: k for k, v in ocfg.class_names.items()}
    targets = {}
    for key, val in ocfg.targets_per_class.items():
        cid = key if isinstance(key, int) else name_to_id.get(key)
        if cid is None:
            print(f"  [!] cible ignorée : clé inconnue {key!r} (vérifiez class_names).")
            continue
        targets[cid] = int(val)
    if ocfg.target_ratio and before:
        max_count = max(before.values())
        for cid in before:
            targets.setdefault(cid, int(math.ceil(max_count / ocfg.target_ratio)))
    return targets


def compute_class_weights(counts, ocfg):
    """Poids de classe pour une focal loss / BCE pondérée (calculés sur la distribution fournie).
    Renvoie deux variantes, ORDONNÉES PAR id de classe (prêtes à passer à une perte) :
      - inverse-fréquence, normalisée (moyenne = 1) ;
      - 'effective number of samples' (Cui et al., 2019), normalisée.
    L'effective-number est plus douce quand une classe est extrêmement rare (évite des poids démesurés)."""
    ids = sorted(ocfg.class_names)
    n = np.array([max(1, counts.get(i, 0)) for i in ids], dtype=np.float64)
    inv = n.sum() / (len(ids) * n)
    inv = inv / inv.mean()
    beta = ocfg.beta_effective
    eff = (1.0 - beta) / (1.0 - np.power(beta, n))
    eff = eff / eff.mean()
    return ids, inv, eff


def build_rare_pool(items, rare_class_ids):
    """Liste de (img_path, box_n, class) pour le Copy-Paste."""
    rare = set(rare_class_ids)
    pool = []
    for p, _, boxes_n, labels in items:
        for b, l in zip(boxes_n, labels):
            if int(l) in rare:
                pool.append((p, b.copy(), int(l)))
    return pool


def copy_paste(img, boxes_xyxy, labels, rare_pool, want_class, min_size, n_max=3):
    """Colle 1..n_max instances de la classe `want_class` (prélevées dans le pool) sur l'image."""
    candidates = [x for x in rare_pool if x[2] == want_class]
    if not candidates:
        return img, boxes_xyxy, labels
    h, w = img.shape[:2]
    nb, nl = list(boxes_xyxy), list(labels)
    for _ in range(random.randint(1, n_max)):
        src_path, box_n, lbl = random.choice(candidates)
        src = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
        sh, sw = src.shape[:2]
        sx1, sy1, sx2, sy2 = yolo_to_xyxy(box_n[None, :], sw, sh)[0].astype(int)
        sx1, sy1, sx2, sy2 = max(0, sx1), max(0, sy1), min(sw, sx2), min(sh, sy2)
        if sx2 - sx1 < min_size or sy2 - sy1 < min_size:
            continue
        patch = src[sy1:sy2, sx1:sx2]
        ph, pw = patch.shape[:2]
        if pw >= w or ph >= h:
            continue
        px, py = random.randint(0, w - pw - 1), random.randint(0, h - ph - 1)
        img[py:py + ph, px:px + pw] = patch
        nb.append(np.array([px, py, px + pw, py + ph], dtype=np.float32))
        nl.append(lbl)
    boxes_xyxy = np.array(nb, np.float32) if nb else np.zeros((0, 4), np.float32)
    labels = np.array(nl, np.int64) if nl else np.zeros((0,), np.int64)
    return img, boxes_xyxy, labels


def write_sample(out_img_dir, out_lbl_dir, name, img, boxes_n, labels):
    cv2.imwrite(os.path.join(out_img_dir, name + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    lp = os.path.join(out_lbl_dir, name + ".txt")
    if len(boxes_n):
        rows = np.concatenate([labels.reshape(-1, 1).astype(np.float32), boxes_n], axis=1)
        np.savetxt(lp, rows, fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])
    else:
        open(lp, "w").close()

def clip_bboxes(bboxes):
    """Remet les bboxes dans [0, 1]"""
    bboxes = np.array(bboxes)
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, 1)  # x_min, x_max
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, 1)  # y_min, y_max
    return bboxes


def generate(ocfg: OfflineConfig, acfg: AugConfig):
    random.seed(ocfg.seed)
    np.random.seed(ocfg.seed)
    os.makedirs(ocfg.out_images_dir, exist_ok=True)
    os.makedirs(ocfg.out_labels_dir, exist_ok=True)

    items = load_dataset(ocfg.images_dir, ocfg.labels_dir)
    before = count_instances(items)
    if not before:
        raise RuntimeError("Aucune annotation trouvée.")

    # 1) recopier le jeu d'origine dans la sortie (jeu enrichi complet)
    if ocfg.copy_originals:
        for p, lp, _, _ in items:
            shutil.copy(p, os.path.join(ocfg.out_images_dir, os.path.basename(p)))
            if os.path.exists(lp):
                shutil.copy(lp, os.path.join(ocfg.out_labels_dir, os.path.basename(lp)))

    # 2) cibles EXPLICITES par classe (avec repli éventuel sur target_ratio)
    target = resolve_targets(ocfg, before)
    to_add = {c: max(0, int(math.ceil(target[c] - before.get(c, 0)))) for c in target}
    rare_ids = tuple(c for c, n in to_add.items() if n > 0)
    names = ocfg.class_names
    print("Instances AVANT :", {names.get(k, k): v for k, v in sorted(before.items())})
    print("À ajouter       :", {names.get(k, k): v for k, v in sorted(to_add.items()) if v > 0})

    acfg.rare_class_ids = rare_ids
    rare_pool = build_rare_pool(items, rare_ids)
    geo_photo = build_geometric_photometric(acfg)
    s = acfg.img_size

    after = Counter(before)  # on cumule au fur et à mesure
    gen_index = 0
    for c in rare_ids:
        # images sources contenant la classe c
        sources = [it for it in items if c in it[3].tolist()]
        if not sources:
            print(f"  [{ocfg.class_names.get(c, c)}] aucune image source, on saute.")
            continue
        produced = 0
        attempts = 0
        while after[c] < target[c] and produced < ocfg.max_generated_per_class and attempts < ocfg.max_generated_per_class * 4:
            attempts += 1
            p, _, boxes_n, labels = random.choice(sources)
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            boxes_xyxy = yolo_to_xyxy(boxes_n, w, h)

            # Copy-Paste : on injecte des instances supplémentaires de la classe c
            if rare_pool and random.random() < acfg.copy_paste_prob:
                img, boxes_xyxy, labels = copy_paste(img, boxes_xyxy, labels, rare_pool, c, acfg.min_box_size_px)

            # géométrie + photométrie (boîtes préservées par Albumentations)
            boxes_n2 = xyxy_to_yolo(boxes_xyxy, img.shape[1], img.shape[0])
            
            out = geo_photo(image=img, bboxes=boxes_n2.tolist(), class_labels=labels.tolist())
            aug_img = out["image"]
            aug_boxes_n = np.array(out["bboxes"], np.float32).reshape(-1, 4)
            aug_labels = np.array(out["class_labels"], np.int64)
            #aug_boxes_n = clip_bboxes(aug_boxes_n) 

            # garde-fou : on jette les boîtes devenues trop petites
            bx = yolo_to_xyxy(aug_boxes_n, s, s)
            bx, aug_labels = filter_small_boxes(bx, aug_labels, acfg.min_box_size_px)
            aug_boxes_n = xyxy_to_yolo(bx, s, s)

            n_c = int((aug_labels == c).sum())
            if n_c == 0:
                continue  # l'augmentation a fait disparaître la classe visée -> on recommence

            write_sample(ocfg.out_images_dir, ocfg.out_labels_dir, f"aug_c{c}_{gen_index:06d}",
                         aug_img, aug_boxes_n, aug_labels)
            gen_index += 1
            produced += 1
            after.update(aug_labels.tolist())
        print(f"  [{ocfg.class_names.get(c, c)}] {produced} images générées "
              f"-> instances : {before.get(c, 0)} -> {after[c]}")

    return before, after


def report(before, after, ocfg):
    out_dir = ocfg.out_images_dir
    names = ocfg.class_names
    classes = sorted(set(before) | set(after) | set(names))
    mb_before = max(before.values()) if before else 1
    ma_after = max(after.values()) if after else 1
    lines = ["classe,avant,apres,ratio_avant,ratio_apres"]
    for c in classes:
        b, a = before.get(c, 0), after.get(c, 0)
        lines.append(f"{names.get(c, c)},{b},{a},{mb_before / max(b,1):.1f},{ma_after / max(a,1):.1f}")
    csv_path = os.path.join(out_dir, "distribution_report.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(lines))
    print("\nRapport distribution :", csv_path)
    print("\n".join(lines))

    # --- poids de classe, calculés sur la distribution APRÈS (celle sur laquelle on entraîne) ---
    ids, inv, eff = compute_class_weights(after, ocfg)
    wlines = ["id,classe,instances_apres,poids_inv_freq,poids_effective_num"]
    for i, idc in enumerate(ids):
        wlines.append(f"{idc},{names.get(idc, idc)},{after.get(idc, 0)},{inv[i]:.4f},{eff[i]:.4f}")
    w_csv = os.path.join(out_dir, "class_weights.csv")
    with open(w_csv, "w") as f:
        f.write("\n".join(wlines))
    print("\nPoids de classe :", w_csv)
    print("  ordre = id de classe", [names.get(i, i) for i in ids])
    print("  inverse-fréquence :", [round(float(x), 3) for x in inv])
    print("  effective-number  :", [round(float(x), 3) for x in eff])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [str(names.get(c, c)) for c in classes]
        x = np.arange(len(classes)); wd = 0.4
        plt.figure(figsize=(11, 5))
        plt.bar(x - wd / 2, [before.get(c, 0) for c in classes], wd, label="avant")
        plt.bar(x + wd / 2, [after.get(c, 0) for c in classes], wd, label="après")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("nombre d'instances")
        plt.title("Distribution des instances avant / après rééquilibrage")
        plt.legend(); plt.tight_layout()
        png = os.path.join(out_dir, "distribution_report.png")
        plt.savefig(png, dpi=120)
        print("Graphique :", png)
    except Exception as e:
        print("(graphique ignoré :", e, ")")


if __name__ == "__main__":
    ocfg = OfflineConfig()           # <-- adaptez chemins, class_names et targets_per_class.
    acfg = AugConfig(img_size=640,rare_class_ids=(2,5,6))   # garde-fous prudents par défaut.
    before, after = generate(ocfg, acfg)
    report(before, after, ocfg)