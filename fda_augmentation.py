"""
Fourier Domain Adaptation (FDA) — Yang & Soatto, CVPR 2020.

Transfère le "style" de bas niveau (couleur, éclairage, texture capteur) d'un
domaine CIBLE (terrain, non annoté) vers une image SOURCE (labo, annotée),
sans déplacer les objets. Les bounding boxes de la source restent donc valides.

Usage typique : augmentation à la volée pendant l'entraînement du détecteur.
Dépendances : numpy, opencv-python  (pip install numpy opencv-python)
"""

import numpy as np
import cv2
import os
import glob
import random
import shutil
 

def _low_freq_mutate(amp_src, amp_trg, beta=0.01):
    """Remplace la fenêtre basse fréquence de l'amplitude source par celle de la cible.

    amp_src, amp_trg : amplitude FFT, shape (C, H, W)
    beta : fraction de la plus petite dimension définissant la taille de la fenêtre.
    """
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))  # basses fréquences au centre
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = int(np.floor(min(h, w) * beta))  # demi-taille de la fenêtre
    c_h, c_w = h // 2, w // 2

    h1, h2 = c_h - b, c_h + b + 1
    w1, w2 = c_w - b, c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]  # le swap
    return np.fft.ifftshift(a_src, axes=(-2, -1))


def fda_source_to_target(src_img, trg_img, beta=0.01):
    """Applique FDA. src_img/trg_img : numpy HxWxC, RGB ou BGR, uint8 ou float.

    Retourne une image uint8 (même taille que src_img) au style de trg_img.
    """
    # On travaille en (C, H, W)
    src = src_img.astype(np.float32).transpose(2, 0, 1)
    trg = trg_img.astype(np.float32)

    # La cible doit avoir la même taille que la source pour le swap d'amplitude
    if trg.shape[:2] != src_img.shape[:2]:
        trg = cv2.resize(trg, (src_img.shape[1], src_img.shape[0]))
    trg = trg.transpose(2, 0, 1)

    fft_src = np.fft.fft2(src, axes=(-2, -1))
    fft_trg = np.fft.fft2(trg, axes=(-2, -1))

    amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
    amp_trg = np.abs(fft_trg)

    # On garde la PHASE source (= la géométrie), on emprunte l'amplitude basse fréquence cible
    amp_mutated = _low_freq_mutate(amp_src.copy(), amp_trg, beta=beta)

    fft_mutated = amp_mutated * np.exp(1j * pha_src)
    src_in_trg = np.real(np.fft.ifft2(fft_mutated, axes=(-2, -1)))

    src_in_trg = src_in_trg.transpose(1, 2, 0)  # retour HxWxC
    return np.clip(src_in_trg, 0, 255).astype(np.uint8)


def fda_multiband(src_img, trg_img, betas=(0.01, 0.03, 0.05)):
    """Variante multi-bandes : moyenne plusieurs beta (plus robuste, moins d'artefacts)."""
    outs = [fda_source_to_target(src_img, trg_img, b).astype(np.float32) for b in betas]
    return np.clip(np.mean(outs, axis=0), 0, 255).astype(np.uint8)


# --- Intégration dans une pipeline d'entraînement (pseudo-code) ---------------
#
# import random, glob
# target_pool = glob.glob("images_terrain_non_annotees/*.jpg")  # pas besoin de labels
#
# def augment(image, boxes):           # image source annotée
#     if random.random() < 0.5:        # appliquer ~50% du temps pour garder de la diversité
#         trg = cv2.imread(random.choice(target_pool))
#         image = fda_source_to_target(image, trg, beta=random.uniform(0.01, 0.05))
#     return image, boxes              # boxes INCHANGÉES : FDA ne déplace rien
#
# Puis chaîne avec tes augmentations photométriques légères + flip/scale habituelles.






 
 
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
 
 
def list_images(d):
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if os.path.splitext(f)[1] in IMG_EXTS
    )
 
 
def labels_have_rare(label_path, rare_set):
    """True si le fichier label YOLO contient au moins une classe rare."""
    if not (os.path.exists(label_path) and os.path.getsize(label_path) > 0):
        return False
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if parts and int(float(parts[0])) in rare_set:
                return True
    return False

def labels_have_rare_only(label_path, rare_set):
    """True si le fichier label YOLO contient au moins une classe rare."""
    if not (os.path.exists(label_path) and os.path.getsize(label_path) > 0):
        return False
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if not (parts and int(float(parts[0])) in rare_set):
                return False
    return True
 
 
def fda_augment_rare(
    source_dir,
    source_label_dir,
    target_dir,
    output_dir,
    output_label_dir,
    beta=[0.001,0.01],
    rare_class_ids=(),
    n_aug_per_image=1,     # nombre d'images FDA par image rare (cible aléatoire à chaque fois)
    match_size=True,       # redimensionne la cible à la taille de la source (sûr pour FDA)
    seed=0,
    au_suffix="_fda_aug",
    copy_source=True, 
    intersection_only=False, 
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    rare_set = {int(c) for c in rare_class_ids}
 
    target_imgs = list_images(target_dir)
    if not target_imgs:
        raise RuntimeError(f"Aucune image cible trouvée dans {target_dir}")
 
    source_imgs = list_images(source_dir)
    if not source_imgs:
        raise RuntimeError(f"Aucune image source trouvée dans {source_dir}")
 
    copied = augmented = skipped = 0
 
    for src_path in source_imgs:
        fname = os.path.basename(src_path)
        stem, ext = os.path.splitext(fname)
        label_name = stem + ".txt"
        src_label = os.path.join(source_label_dir, label_name)
 
        # 1) recopier l'original (image + label) vers la sortie
        if copy_source:
            shutil.copy2(src_path, os.path.join(output_dir, fname))
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(output_label_dir, label_name))
            copied += 1
 
        # 2) FDA seulement si l'image contient une classe rare
        if intersection_only:
            if not labels_have_rare_only(src_label, rare_set):
                continue
        
        if not intersection_only:
            if not labels_have_rare(src_label, rare_set):
                continue
 
        src_img = cv2.imread(src_path)
        if src_img is None:
            print(f"  [!] lecture impossible : {src_path}")
            skipped += 1
            continue
 
        for k in range(n_aug_per_image):
            trg_path = random.choice(target_imgs)
            trg_img = cv2.imread(trg_path)
            if trg_img is None:
                skipped += 1
                continue
            if match_size:
                trg_img = cv2.resize(trg_img, (src_img.shape[1], src_img.shape[0]))
 
            try:
                out = fda_source_to_target(src_img, trg_img, beta=random.uniform(beta[0], beta[1]))
            except Exception as e:
                print(f"  [!] FDA échouée sur {fname} : {e}")
                skipped += 1
                continue
 
            # FDA peut renvoyer du float -> on borne en [0,255] uint8 avant l'écriture
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
 
            suffix = au_suffix if n_aug_per_image == 1 else f"{au_suffix}{k}"
            out_img_name = f"{stem}{suffix}{ext}"
            out_lbl_name = f"{stem}{suffix}.txt"
 
            cv2.imwrite(os.path.join(output_dir, out_img_name), out)
            # même géométrie -> on copie le label source sous le nouveau nom
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(output_label_dir, out_lbl_name))
            else:
                open(os.path.join(output_label_dir, out_lbl_name), "w").close()
            augmented += 1
 
    print(f"Originaux copiés  : {copied}")
    print(f"Images FDA créées : {augmented}")
    print(f"Ignorées          : {skipped}")
    return augmented



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python fda_augmentation.py source.jpg target.jpg sortie.jpg")
        sys.exit(1)
    src = cv2.imread(sys.argv[1])
    trg = cv2.imread(sys.argv[2])
    out = fda_source_to_target(src, trg, beta=0.03)
    cv2.imwrite(sys.argv[3], out)
    print("FDA appliquée ->", sys.argv[3])
