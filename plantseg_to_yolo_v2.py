"""
Conversion PlantSeg → YOLO v2
Corrections:
  - Gère category_id=0 (PlantSeg commence à 0, pas 1)
  - Fallback si "categories" absent du JSON (lecture depuis MetadataV2.csv)
  - Ignore les annotations dont category_id n'est pas dans le map
  - Rapport de conversion à la fin

Usage:
    python plantseg_to_yolo_v2.py --mode detect   --src ./plantseg --dst ./yolo_detect
    python plantseg_to_yolo_v2.py --mode segment  --src ./plantseg --dst ./yolo_seg
"""

import json
import argparse
import shutil
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
import cv2


# ──────────────────────────────────────────────────────────────
# 1. Chargement JSON + mappage classes (robuste)
# ──────────────────────────────────────────────────────────────

def load_coco_json(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def build_class_map(coco_data: dict, metadata_csv: Path = None) -> tuple:
    """
    Construit le mappage category_id → yolo_class_id (0-based).

    PlantSeg peut avoir:
      - category_id commençant à 0  (pas standard COCO)
      - category_id commençant à 1  (standard COCO)
      - "categories" absent du JSON → fallback sur MetadataV2.csv
    """
    categories = coco_data.get("categories", [])

    # ── Fallback: reconstruire depuis les annotations ──
    if not categories:
        print(" .. Clé 'categories' absente — reconstruction depuis les annotations")
        ids_seen = sorted(set(a["category_id"] for a in coco_data.get("annotations", [])))
        # Essai depuis MetadataV2.csv pour avoir les vrais noms
        names_from_csv = {}
        #metadata_csv=Path("/content/plantseg/plantsegv3/Metadatav2.csv")
        if metadata_csv and metadata_csv.exists():
            with open(metadata_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                print(reader)
                for row in reader:
                    # Cherche une colonne qui ressemble à un id de maladie
                    k ="Index"
                    #print(k)
                    #print(row)
                    if k in row:
                        try:
                            names_from_csv[int(row[k])] = row.get("Disease", row.get("label", row.get("name", str(row[k]))))
                            #print(names_from_csv[int(row[k])])
                        except ValueError:
                            pass
        categories = [
            {"id": cid, "name": names_from_csv.get(cid, f"disease_{cid}")}
            for cid in ids_seen
        ]
        print(f"  → {len(categories)} catégories reconstruites")

    # ── Tri par id ──
    categories = sorted(categories, key=lambda c: c["id"])
    min_id = categories[0]["id"]
    if min_id == 0:
        print(f"  ℹ category_id commence à 0 (mode PlantSeg natif)")
    elif min_id == 1:
        print(f"  ℹ category_id commence à 1 (mode COCO standard)")
    else:
        print(f"  ⚠ category_id commence à {min_id} (inhabituel)")

    # yolo_class_id = position dans la liste triée (toujours 0-based)
    cat_id_to_yolo_id = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]

    return cat_id_to_yolo_id, class_names


# ──────────────────────────────────────────────────────────────
# 2. YOLO Détection
# ──────────────────────────────────────────────────────────────

def bbox_coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    # Clamp pour éviter les valeurs hors image
    cx = max(0.0, min(1.0, (x + w/2) / img_w))
    cy = max(0.0, min(1.0, (y + h/2) / img_h))
    bw = max(0.0, min(1.0, w / img_w))
    bh = max(0.0, min(1.0, h / img_h))
    return cx, cy, bw, bh


def convert_detection(coco_data, cat_map, label_dir: Path):
    label_dir.mkdir(parents=True, exist_ok=True)
    images = {img["id"]: img for img in coco_data["images"]}
    anns_by_image = {}
    skipped = 0
    for ann in coco_data["annotations"]:
        cid = ann["category_id"]
        if cid not in cat_map:
            skipped += 1
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    if skipped:
        print(f"  ⚠ {skipped} annotations ignorées (category_id inconnu)")

    for img_id, info in images.items():
        w, h = info["width"], info["height"]
        stem = Path(info["file_name"]).stem
        lines = []
        for ann in anns_by_image.get(img_id, []):
            cls = cat_map[ann["category_id"]]
            cx, cy, bw, bh = bbox_coco_to_yolo(ann["bbox"], w, h)
            if bw > 0 and bh > 0:
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (label_dir / f"{stem}.txt").write_text("\n".join(lines))

    print(f"  ✓ {len(images)} labels détection → {label_dir}")


# ──────────────────────────────────────────────────────────────
# 3. YOLO Segmentation
# ──────────────────────────────────────────────────────────────

def polygon_to_yolo(seg_flat, img_w, img_h):
    out = []
    for i in range(0, len(seg_flat) - 1, 2):
        out.append(max(0.0, min(1.0, seg_flat[i]   / img_w)))
        out.append(max(0.0, min(1.0, seg_flat[i+1] / img_h)))
    return out


def mask_to_polygons(mask_path: Path, min_area=50):
    mask = np.array(Image.open(mask_path).convert("L"))
    h, w = mask.shape
    results = []
    for val in np.unique(mask):
        if val == 0:
            continue
        binary = (mask == val).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            flat = cnt.flatten().tolist()
            if len(flat) < 6:
                continue
            norm = []
            for i in range(0, len(flat) - 1, 2):
                norm.append(flat[i] / w)
                norm.append(flat[i+1] / h)
            results.append((int(val), norm))   # val brut; mappage fait dans convert_seg
    return results


def convert_segmentation(coco_data, cat_map, ann_dir: Path, label_dir: Path, use_masks=False):
    label_dir.mkdir(parents=True, exist_ok=True)
    images = {img["id"]: img for img in coco_data["images"]}
    anns_by_image = {}
    skipped = 0
    for ann in coco_data["annotations"]:
        cid = ann["category_id"]
        if cid not in cat_map:
            skipped += 1
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    if skipped:
        print(f"  ⚠ {skipped} annotations ignorées (category_id inconnu)")

    for img_id, info in images.items():
        w, h = info["width"], info["height"]
        stem = Path(info["file_name"]).stem
        lines = []

        if use_masks:
            mask_path = ann_dir / (stem + ".png")
            if mask_path.exists():
                for raw_val, poly in mask_to_polygons(mask_path):
                    # Dans les masques PlantSeg: valeur pixel = category_id
                    if raw_val in cat_map:
                        cls = cat_map[raw_val]
                    else:
                        cls = raw_val - 1   # fallback 0-based
                    coords = " ".join(f"{v:.6f}" for v in poly)
                    lines.append(f"{cls} {coords}")
        else:
            for ann in anns_by_image.get(img_id, []):
                if not ann.get("segmentation"):
                    continue
                cls = cat_map[ann["category_id"]]
                for seg in ann["segmentation"]:
                    if isinstance(seg, list):
                        norm = polygon_to_yolo(seg, w, h)
                    else:
                        continue   # RLE non supporté ici
                    if len(norm) >= 6:
                        coords = " ".join(f"{v:.6f}" for v in norm)
                        lines.append(f"{cls} {coords}")

        (label_dir / f"{stem}.txt").write_text("\n".join(lines))

    print(f"  ✓ {len(images)} labels segmentation → {label_dir}")


# ──────────────────────────────────────────────────────────────
# 4. Copie images + dataset.yaml
# ──────────────────────────────────────────────────────────────

def copy_images(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for f in src.glob(ext):
            shutil.copy2(f, dst / f.name)
            n += 1
    print(f"  ✓ {n} images copiées → {dst}")
    return n


def write_yaml(dst: Path, class_names: list, mode: str):
    data = {
        "path":  str(dst.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
        "task":  mode,
    }
    p = dst / "dataset.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"  ✓ dataset.yaml ({len(class_names)} classes) → {p}")
    return p


# ──────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",  default="./plantseg")
    parser.add_argument("--dst",  default="./yolo_dataset")
    parser.add_argument("--mode", choices=["detect", "segment"], default="detect")
    parser.add_argument("--use-masks", action="store_true")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    metadata_csv = src / "Metadatav2.csv"
    class_names = None

    # ── Inspection rapide du JSON train pour détecter le format ──
    train_json = src / "annotation_train.json"
    if train_json.exists():
        sample = load_coco_json(train_json)
        print("── INSPECTION annotation_train.json ──")
        print(f"  clés racine      : {list(sample.keys())}")
        cats = sample.get("categories", [])
        print(f"  nb categories    : {len(cats)}")
        if cats:
            ids = [c["id"] for c in cats]
            print(f"  category_id range: {min(ids)} → {max(ids)}")
        anns = sample.get("annotations", [])
        if anns:
            cids = set(a["category_id"] for a in anns)
            print(f"  annotation cat_id range: {min(cids)} → {max(cids)}")
        print()

    for split in ["train", "val", "test"]:
        print(f"── {split.upper()} ──")
        json_path = src / f"annotation_{split}.json"
        if not json_path.exists():
            print(f"  ⚠ {json_path} introuvable, ignoré.\n")
            continue

        coco = load_coco_json(json_path)
        cat_map, names = build_class_map(coco, Path(metadata_csv))

        if class_names is None:
            class_names = names
        print(f"  {len(names)} classes")

        copy_images(src / "images" / split, dst / split / "images")

        label_dir = dst / split / "labels"
        if args.mode == "detect":
            convert_detection(coco, cat_map, label_dir)
        else:
            ann_dir = src / "annotations" / split
            convert_segmentation(coco, cat_map, ann_dir, label_dir, args.use_masks)
        print()

    print("── YAML ──")
    yaml_path = write_yaml(dst, class_names or [], args.mode)

    suffix = "seg" if args.mode == "segment" else ""
    print(f"\n✅ Terminé.")
    print(f"   yolo train model=yolo11n{suffix}.pt data={yaml_path} epochs=50 imgsz=640")


if __name__ == "__main__":
    main()
