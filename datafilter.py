import os
import shutil
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def count_dataset_stats(base_path, classes, split='train'):
    """Compte les images et labels par classe dans un split YOLO."""
    img_dir   = Path(base_path) / split / 'images'
    label_dir = Path(base_path) / split / 'labels'
    class_counts = {i: 0 for i in range(len(classes))}

    if not label_dir.exists():
        print(f'⚠️  Dossier labels introuvable : {label_dir}')
        return class_counts # Return empty counts if dir not found

    label_files = list(label_dir.glob('*.txt'))
    for lf in label_files:
        with open(lf) as f:
            for line in f:
                cls_id = int(float(line.strip().split()[0]))
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    total_imgs   = len(list(img_dir.glob('*.*')))
    total_labels = sum(class_counts.values())
    print(f'\n📊 {split.upper()} — {total_imgs:,} images | {total_labels:,} annotations')
    print(f'{"Classe":<35} {"Instances":>10}')
    print('-' * 47)
    for idx, name in enumerate(classes):
        print(f'{name:<35} {class_counts[idx]:>10,}')
    return class_counts

def filter_and_remap_dataset(source_base, dest_base, split, id_map):
    """
    Filters and re-maps YOLO labels for specified classes, copying corresponding images.
    """
    source_images_dir = os.path.join(source_base, split, 'images')
    source_labels_dir = os.path.join(source_base, split, 'labels')
    dest_images_dir = os.path.join(dest_base, split, 'images')
    dest_labels_dir = os.path.join(dest_base, split, 'labels')

    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)

    print(f"\nProcessing {split} split...")
    processed_files_count = 0
    copied_images_count = 0
    filtered_annotations_count = 0

    if not os.path.exists(source_labels_dir):
        print(f"Source labels directory not found: {source_labels_dir}")
        return

    for label_filename in os.listdir(source_labels_dir):
        if label_filename.endswith('.txt'):
            source_label_path = os.path.join(source_labels_dir, label_filename)

            image_filename_base = os.path.splitext(label_filename)[0]
            found_image = False
            source_image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                current_image_path = os.path.join(source_images_dir, image_filename_base + ext)
                if os.path.exists(current_image_path):
                    source_image_path = current_image_path
                    found_image = True
                    break

            if not found_image:
                # print(f"Warning: No corresponding image found for {label_filename} in {source_images_dir}. Skipping.")
                continue

            filtered_lines = []
            with open(source_label_path, 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        try:
                            original_id = int(float(parts[0]))
                            if original_id in id_map:
                                new_id = id_map[original_id]
                                filtered_line = f"{new_id} {' '.join(parts[1:])}\n"
                                filtered_lines.append(filtered_line)
                                filtered_annotations_count += 1
                        except ValueError:
                            print(f"Warning: Could not parse class ID in line: {line.strip()} in file {label_filename}. Skipping line.")
                            continue

            if filtered_lines:
                dest_label_path = os.path.join(dest_labels_dir, label_filename)
                with open(dest_label_path, 'w') as f_out:
                    f_out.writelines(filtered_lines)
                shutil.copy(source_image_path, os.path.join(dest_images_dir, os.path.basename(source_image_path)))
                copied_images_count += 1
                processed_files_count += 1

    print(f"Finished processing {split} split.")
    print(f"  {processed_files_count} label files processed and copied.")
    print(f"  {copied_images_count} images copied.")
    print(f"  {filtered_annotations_count} annotations for target classes found.")

def main():
    # Define CONFIG and other parameters
    CONFIG = {
        'dataset_base': '/content/Tomato-Village/Variant-c(Object Detection)',
        'yaml_path': 'tomato_village_obj_det.yaml',
        'results_dir': '/content/Tomato-Village/results',
        'class_names': [
            'Early_blight', 'Healthy', 'Late_blight', 'Leaf Miner',
            'Magnesium Deficiency', 'Nitrogen Deficiency',
            'Pottassium Deficiency', 'Spotted Wilt Virus'
        ],
    }
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    source_base_path = '/content/Tomato-Village/Variant-c(Object Detection)'
    dest_base_path = '/content/Tomato-Village/detection' # New dest path

    original_class_names = CONFIG['class_names']
    target_classes_map = {
        'Late_blight': 0,
        'Spotted Wilt Virus': 1
    }

    # Process the dataset
    os.makedirs(dest_base_path, exist_ok=True)

    original_to_new_id_map = {}
    for class_name, new_id in target_classes_map.items():
        if class_name in original_class_names:
            original_id = original_class_names.index(class_name)
            original_to_new_id_map[original_id] = new_id
        else:
            print(f"Warning: Target class '{class_name}' not found in original_class_names.")

    print(f"Original to New ID Map: {original_to_new_id_map}")

    filter_and_remap_dataset(source_base_path, dest_base_path, 'train', original_to_new_id_map)
    filter_and_remap_dataset(source_base_path, dest_base_path, 'val', original_to_new_id_map)

    yaml_data = {
        'path': dest_base_path,
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(target_classes_map),
        'names': list(target_classes_map.keys())
    }

    yaml_path = os.path.join(dest_base_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\nCreated new data.yaml at: {yaml_path}")
    print(yaml.dump(yaml_data, sort_keys=False))

    # Verify and Visualize
    print("\nVerifying new dataset structure and counts:")

    FILTERED_CONFIG = {
        'dataset_base': dest_base_path,
        'class_names': list(target_classes_map.keys())
    }

    print("Counts for TRAIN split of filtered dataset:")
    filtered_train_counts = count_dataset_stats(FILTERED_CONFIG['dataset_base'], FILTERED_CONFIG['class_names'], 'train')
    print("\nCounts for VAL split of filtered dataset:")
    filtered_val_counts = count_dataset_stats(FILTERED_CONFIG['dataset_base'], FILTERED_CONFIG['class_names'], 'val')

    if filtered_train_counts and filtered_val_counts:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, (counts, title) in zip(
            axes,
            [(filtered_train_counts, 'Train Filtered'), (filtered_val_counts, 'Validation Filtered')]
        ):
            names  = FILTERED_CONFIG['class_names']
            values = [counts.get(i, 0) for i in range(len(names))]
            colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
            bars = ax.bar(range(len(names)), values, color=colors)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
            ax.set_title(f'Distribution classes — {title}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Nb instances')
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(val), ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(dest_base_path, 'filtered_class_distribution.png'), dpi=150)
        # plt.show() # Comment out plt.show() for non-interactive execution

if __name__ == '__main__':
    main()
