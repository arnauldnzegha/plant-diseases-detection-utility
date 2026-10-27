"""
Conversion d'un dataset au format YOLO (data.yaml + labels .txt) vers le format
COCO (annotations.json), pour l'entraînement de RT-DETRv2 (HuggingFace Transformers
ou depot officiel lyuwenyu/RT-DETR).

Hypothese sur la structure (convention standard Ultralytics/YOLO) :
    <path>/
        train/images/*.jpg (ou .png)
        train/labels/*.txt   <- meme nom de base que l'image correspondante
        val/images/...
        val/labels/...
        test/images/...
        test/labels/...

Chaque fichier .txt contient une ligne par objet :
    class_id x_center y_center width height
(valeurs normalisees entre 0 et 1, comme dans le format YOLO standard)

Usage :
    python yolo2coco.py --yaml /chemin/vers/dataset.yaml --split train
    python yolo2coco.py --yaml /chemin/vers/dataset.yaml --split val
    python yolo2coco.py --yaml /chemin/vers/dataset.yaml --split test

Cela genere un fichier <split>_annotations.json a la racine de "path" (dans dataset.yaml).

IMPORTANT : je n'ai pas eu accès à vos images/labels réels pour tester ce script
(seul dataset.yaml a été fourni). Il a été testé avec un jeu de données factice
reproduisant la structure attendue (cf. section de test dans la conversation).
Vérifiez la sortie sur un petit échantillon de VOS données avant de lancer
la conversion complète.
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def load_dataset_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def find_images(images_dir):
    images_dir = Path(images_dir)
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(sorted(images_dir.glob(f"*{ext}")))
    return files


def yolo_txt_to_boxes(txt_path):
    """Lit un fichier .txt YOLO et retourne une liste de (class_id, xc, yc, w, h)
    normalisees. Retourne une liste vide si le fichier n'existe pas (image sans
    objet, cas valide en detection)."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Ligne malformee dans {txt_path} (ligne {line_num}): '{line}'"
                )
            class_id, xc, yc, w, h = parts
            boxes.append(
                (int(float(class_id)), float(xc), float(yc), float(w), float(h))
            )
    return boxes


def convert_split(dataset_root, images_rel_path, class_names, split_name):
    images_dir = Path(dataset_root) / images_rel_path
    # convention standard : labels/ est un dossier "frere" de images/
    labels_dir = Path(str(images_dir).replace(
        os.sep + "images", os.sep + "labels"
    ))
    if not images_dir.exists():
        raise FileNotFoundError(f"Dossier d'images introuvable : {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(
            f"Dossier de labels introuvable : {labels_dir} "
            "(attendu comme dossier frere de images/, convention YOLO standard - "
            "verifiez la structure reelle de votre dataset si ce chemin est different)"
        )

    image_files = find_images(images_dir)
    if not image_files:
        raise FileNotFoundError(f"Aucune image trouvee dans {images_dir}")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name} for i, name in enumerate(class_names)
        ],
    }

    ann_id = 1
    skipped_unreadable = []

    for img_id, img_path in enumerate(image_files, start=1):
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception as e:
            skipped_unreadable.append((str(img_path), str(e)))
            continue

        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        txt_path = labels_dir / (img_path.stem + ".txt")
        boxes = yolo_txt_to_boxes(txt_path)

        for class_id, xc, yc, w, h in boxes:
            # conversion des coordonnees normalisees YOLO -> pixels absolus COCO
            box_w = w * width
            box_h = h * height
            x_min = (xc * width) - (box_w / 2)
            y_min = (yc * height) - (box_h / 2)

            # garde-fou : coordonnees negatives ou hors image (peut arriver si
            # une annotation source est legerement mal formee)
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            box_w = min(box_w, width - x_min)
            box_h = min(box_h, height - y_min)

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [round(x_min, 2), round(y_min, 2),
                              round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    if skipped_unreadable:
        print(f"[ATTENTION] {len(skipped_unreadable)} image(s) illisible(s) ignoree(s) :")
        for path, err in skipped_unreadable[:10]:
            print(f"  - {path}: {err}")

    return coco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="Chemin vers dataset.yaml")
    parser.add_argument(
        "--split", required=True, choices=["train", "val", "test"],
        help="Split a convertir",
    )
    parser.add_argument(
        "--output", default=None,
        help="Chemin du JSON de sortie (par defaut : <split>_annotations.json a la racine du dataset)",
    )
    parser.add_argument(
        "--dataset-root", default=None,
        help="Surcharge le champ 'path' du yaml (utile si le dataset a ete deplace)",
    )
    args = parser.parse_args()

    cfg = load_dataset_yaml(args.yaml)

    class_names = cfg["names"]
    if isinstance(class_names, dict):
        # certains data.yaml utilisent {0: 'nom', 1: 'nom', ...} au lieu d'une liste
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    dataset_root = args.dataset_root or cfg["path"]
    images_rel_path = cfg[args.split]

    print(f"Dataset root : {dataset_root}")
    print(f"Split        : {args.split} -> {images_rel_path}")
    print(f"Classes ({len(class_names)}) : {class_names}")

    coco = convert_split(dataset_root, images_rel_path, class_names, args.split)

    output_path = args.output or str(Path(dataset_root) / f"{args.split}_annotations.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"\nOK : {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Ecrit dans : {output_path}")


if __name__ == "__main__":
    main()
