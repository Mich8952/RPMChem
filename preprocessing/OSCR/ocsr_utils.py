import os
import sys
import logging
import warnings
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# --- Path setup ---
MOLNEXTR_REPO_PATH = "/home/rosaliewang/MolNexTR"
if MOLNEXTR_REPO_PATH not in sys.path:
    sys.path.insert(0, MOLNEXTR_REPO_PATH)

# No checkpoint path needed — MolNexTRSingleton handles its own weights via pystow
from MolNexTR import get_predictions, MolNexTRSingleton  # type: ignore


def image_to_smiles(image_input, return_molfile=False):
    """
    Convert a molecular structure image to a SMILES string.
    Uses MolNexTR's get_predictions() API.

    Args:
        image_input: str (file path) or PIL.Image.Image
        return_molfile: if True, also return the predicted MOLfile

    Returns:
        smiles (str) or None if prediction failed
        molfile (str) only if return_molfile=True
    """
    temp_path = None
    if isinstance(image_input, Image.Image):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_input.save(tmp.name)
        temp_path = tmp.name
        image_path = temp_path
    else:
        image_path = str(image_input)

    try:
        result = get_predictions(image_path)
        smiles = result.get("predicted_smiles", None)

        # Canonicalize with RDKit if available
        if smiles:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                else:
                    logger.warning(f"RDKit could not parse SMILES: {smiles} — returning raw")
            except ImportError:
                pass

        if return_molfile:
            return smiles, result.get("predicted_molfile", None)
        return smiles

    except Exception as e:
        logger.warning(f"MolNexTR prediction failed for {image_path}: {e}")
        return (None, None) if return_molfile else None

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def is_likely_molecule(image: Image.Image, min_size: int = 50) -> bool:
    """
    Heuristic: is this image crop likely a molecular structure?
    Filters out arrows, text boxes, plots, etc.
    """
    w, h = image.size
    if w < min_size or h < min_size:
        return False

    # Very wide/tall = likely a text line or reaction arrow, not a structure
    aspect = max(w, h) / min(w, h)
    if aspect > 5.0:
        return False

    gray = np.array(image.convert("L"), dtype=float)

    # Molecule structures: mostly white background, sparse black lines
    white_frac = (gray > 200).mean()
    dark_frac = (gray < 80).mean()

    if white_frac < 0.5:
        return False   # not a white-background image (e.g. photo, filled plot)
    if dark_frac < 0.01 or dark_frac > 0.4:
        return False   # too empty or too dense

    return True
