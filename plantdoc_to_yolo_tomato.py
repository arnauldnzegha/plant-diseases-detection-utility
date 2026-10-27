"""
PlantDoc Dataset → Format YOLO (4 classes communes Tomato-Village)
===================================================================
Filtre uniquement les 4 classes compatibles entre PlantDoc
et ton modèle Tomato-Village, avec des IDs continus (0→3).

Mapping final :
    PlantDoc                    → Tomato-Village        ID
    "Tomato Early blight leaf"  → Early_blight           0
    "Tomato leaf"               → Healthy                1
    "Tomato leaf late blight"   → Late_blight            2
    "Tomato leaf yellow virus"  → Spotted Wilt Virus     3

⚠️  ATTENTION à l'évaluation :
    Ton modèle a été entraîné avec Spotted Wilt Virus = ID 7.
    Dans ce dataset il sera ID 3.
    → Utilise ce data.yaml UNIQUEMENT pour l'évaluation sur PlantDoc,
      pas pour l'inférence en production.

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
OUTPUT_DIR  = Path("./plantdoc_yolo")

# Mapping PlantDoc → nom Tomato-Village (IDs continus 0→3)
CLASS_MAPPING = {
    "Tomato Early blight leaf" : "Early_blight",
    "Tomato leaf"              : "Healthy",
    "Tomato leaf late blight"  : "Late_blight",
    "Tomato leaf yellow virus" : "Spotted Wilt Virus",
}

# Classes finales — ordre = ID YOLO dans ce dataset
TARGET_CLASSES = [
    "Early_blight",       # 0
    "Healthy",            # 1
    "Late_blight",        # 2
    "Spotted Wilt Virus", # 3  ← était 7 dans Tomato-Village
]

CLASS2ID = {c: i for i, c in enumerate(TARGET_CLASSES)}


# ─────────────────────────────────────────────
# CONVERSION CSV → YOLO
# ─────────────────────────────────────────────

def convert_csv(csv_path: Path, img_src_dir: Path,
                img_dst_dir: Path, label_dst_dir: Path):

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={"classname": "class", "label": "class",
                        "image": "filename", "file": "filename"}, inplace=True)

    # Garder uniquement les lignes mappables
    df["mapped_class"] = df["class"].map(CLASS_MAPPING)
    df = df[df["mapped_class"].notna()]

    if df.empty:
        print(f"[WARN] Aucune ligne tomate trouvée dans {csv_path.name}")
        return

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
            mapped = row["mapped_class"]
            cls_id = CLASS2ID[mapped]

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
            stats[mapped] = stats.get(mapped, 0) + 1

        txt_path = label_dst_dir / (img_src.stem + ".txt")
        txt_path.write_text("\n".join(lines))
        converted += 1

    print(f"[OK] {csv_path.name} → {converted} images converties, {skipped} ignorées")
    for cls, count in sorted(stats.items()):
        print(f"     {cls:<25} : {count} annotations")


# ─────────────────────────────────────────────
# GÉNÉRER data.yaml
# ─────────────────────────────────────────────

def generate_yaml(output_dir: Path):
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(f"""\
# PlantDoc — 4 classes communes avec Tomato-Village
# ATTENTION : IDs différents de Tomato-Village pour Spotted Wilt Virus
#   Tomato-Village : Spotted Wilt Virus = ID 7
#   Ce fichier     : Spotted Wilt Virus = ID 3
path: {output_dir.resolve()}

train: TRAIN/images
val:   TRAIN/images
test:  TEST/images

nc: {len(TARGET_CLASSES)}
names: {TARGET_CLASSES}
""")
    print(f"[OK] data.yaml → {yaml_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  PlantDoc → YOLO (4 classes communes)")
    print("=" * 55)

    print("\n[Mapping actif]")
    for src, dst in CLASS_MAPPING.items():
        print(f"  '{src}'  →  '{dst}'  (id={CLASS2ID[dst]})")

    for split in ("TRAIN", "TEST"):
        print(f"\n── {split} ──")
        convert_csv(
            csv_path      = DATASET_DIR / f"{split.lower()}_labels.csv",
            img_src_dir   = DATASET_DIR / split,
            img_dst_dir   = OUTPUT_DIR / split / "images",
            label_dst_dir = OUTPUT_DIR / split / "labels",
        )

    generate_yaml(OUTPUT_DIR)

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