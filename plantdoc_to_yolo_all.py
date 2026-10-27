"""
PlantDoc Dataset → Format YOLO (toutes les classes)
=====================================================
Convertit l'intégralité du dataset PlantDoc en format YOLO.
Les classes sont extraites automatiquement depuis les CSV.

Structure générée :
    plantdoc_yolo/
    ├── TRAIN/
    │   ├── images/
    │   └── labels/
    ├── TEST/
    │   ├── images/
    │   └── labels/
    └── data.yaml

Prérequis :
    pip install pandas
    git clone https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git
"""

import shutil
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATASET_DIR = Path("./PlantDoc-Object-Detection-Dataset")
OUTPUT_DIR  = Path("./plantdoc_yolo_all")


# ─────────────────────────────────────────────
# 1. EXTRAIRE TOUTES LES CLASSES
# ─────────────────────────────────────────────

def get_classes(train_csv: Path, test_csv: Path) -> list[str]:
    df = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    class_col = next(c for c in df.columns if c in ("class", "label", "classname"))
    classes = sorted(df[class_col].dropna().unique().tolist())
    print(f"[INFO] {len(classes)} classes détectées :")
    for i, c in enumerate(classes):
        print(f"   [{i:>2}] {c}")
    return classes


# ─────────────────────────────────────────────
# 2. CONVERSION CSV → YOLO .txt
# ─────────────────────────────────────────────

def convert_csv(csv_path: Path, img_src_dir: Path,
                img_dst_dir: Path, label_dst_dir: Path,
                class2id: dict):

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={"classname": "class", "label": "class",
                        "image": "filename", "file": "filename"}, inplace=True)

    skipped   = 0
    converted = 0
    stats     = {}

    for filename, group in df.groupby("filename"):
        img_src = img_src_dir / filename
        if not img_src.exists():
            candidates = list(img_src_dir.glob(f"{Path(filename).stem}.*"))
            if not candidates:
                skipped += 1
                continue
            img_src = candidates[0]

        # Copier l'image
        img_dst = img_dst_dir / img_src.name
        if not img_dst.exists():
            shutil.copy2(img_src, img_dst)

        # Générer le .txt YOLO
        lines = []
        for _, row in group.iterrows():
            cls_name = str(row["class"]).strip()
            cls_id   = class2id.get(cls_name)
            if cls_id is None:
                continue

            img_w = float(row["width"])
            img_h = float(row["height"])
            xmin  = max(0.0, float(row["xmin"]))
            ymin  = max(0.0, float(row["ymin"]))
            xmax  = min(img_w, float(row["xmax"]))
            ymax  = min(img_h, float(row["ymax"]))

            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue

            cx = (xmin + xmax) / 2 / img_w
            cy = (ymin + ymax) / 2 / img_h
            bw /= img_w
            bh /= img_h

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats[cls_name] = stats.get(cls_name, 0) + 1

        txt_path = label_dst_dir / (img_src.stem + ".txt")
        txt_path.write_text("\n".join(lines))
        converted += 1

    print(f"[OK] {csv_path.name} → {converted} images converties, {skipped} ignorées")
    for cls, count in sorted(stats.items()):
        print(f"     [{class2id[cls]:>2}] {cls:<40} : {count} annotations")


# ─────────────────────────────────────────────
# 3. GÉNÉRER data.yaml
# ─────────────────────────────────────────────

def generate_yaml(output_dir: Path, classes: list[str]):
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(f"""\
# PlantDoc — toutes les classes
path: {output_dir.resolve()}

train: TRAIN/images
val:   TRAIN/images
test:  TEST/images

nc: {len(classes)}
names: {classes}
""")
    print(f"[OK] data.yaml → {yaml_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  PlantDoc → YOLO (toutes les classes)")
    print("=" * 55)

    train_csv = DATASET_DIR / "train_labels.csv"
    test_csv  = DATASET_DIR / "test_labels.csv"

    classes  = get_classes(train_csv, test_csv)
    class2id = {c: i for i, c in enumerate(classes)}

    for split in ("TRAIN", "TEST"):
        print(f"\n── {split} ──")
        convert_csv(
            csv_path      = DATASET_DIR / f"{split.lower()}_labels.csv",
            img_src_dir   = DATASET_DIR / split,
            img_dst_dir   = OUTPUT_DIR / split / "images",
            label_dst_dir = OUTPUT_DIR / split / "labels",
            class2id      = class2id,
        )

    generate_yaml(OUTPUT_DIR, classes)

    print("\n Structure finale :")
    for split in ("TRAIN", "TEST"):
        n_img = len(list((OUTPUT_DIR / split / "images").iterdir()))
        n_lbl = len(list((OUTPUT_DIR / split / "labels").iterdir()))
        print(f"   {split}/images/ ({n_img} images)  "
              f"{split}/labels/ ({n_lbl} fichiers)")
    print(f"   data.yaml")
    print("\n[DONE] ✓")


if __name__ == "__main__":
    main()