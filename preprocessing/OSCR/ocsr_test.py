"""
test_ocsr.py

Manual test: run MolNexTR on a folder of molecule images
and print the predicted SMILES. Use this to sanity-check before
integrating into get_jsons.py.

Usage:
    conda activate molnextr
    python preprocessing/test_ocsr.py --image_dir ./test_molecules/
    python preprocessing/test_ocsr.py --image ./test_molecules/benzene.png
"""

import argparse
import os
from pathlib import Path
from PIL import Image

# must run from RPMChem root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.ocsr_utils import image_to_smiles, is_likely_molecule

def test_single(image_path):
    img = Image.open(image_path).convert("RGB")
    print(f"\nImage: {image_path}")
    print(f"  Size: {img.size}")
    print(f"  is_likely_molecule: {is_likely_molecule(img)}")
    smiles = image_to_smiles(image_path)
    print(f"  Predicted SMILES: {smiles}")
    if smiles:
        # Optionally draw back with rdkit to visually verify
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                out_path = str(image_path).replace(".png", "_rdkit_render.png")
                Draw.MolToFile(mol, out_path, size=(300, 300))
                print(f"  RDKit render saved to: {out_path}")
        except ImportError:
            pass

def test_directory(image_dir):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in extensions]
    print(f"Found {len(paths)} images in {image_dir}")
    for p in sorted(paths):
        test_single(p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    args = parser.parse_args()

    if args.image:
        test_single(args.image)
    elif args.image_dir:
        test_directory(args.image_dir)
    else:
        print("Provide --image or --image_dir. Example:")
        print("  python preprocessing/test_ocsr.py --image ./test_molecules/aspirin.png")
