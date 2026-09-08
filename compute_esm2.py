# ==============================================================================
# Description: Pipeline to compute per-residue ESM-2 embeddings for every
#              protein chain of one DNAproDB JSON entry, storing the
#              result as one .pt file in the output folder.
# Author: Miguel Sanchez Oliveros
# Version: 1.0.0
# Email: miguel.sanchez.oliveros@gmail.com
# ==============================================================================

__author__ = "Miguel Sanchez Oliveros"
__version__ = "1.0.0"
__email__ = "miguel.sanchez.oliveros@gmail.com"



# ==============================================================================
# LOAD LIBRARIES
# ==============================================================================
import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import esm as esm_module

import torch



# ==============================================================================
# CONSTANTS
# ==============================================================================
# Supported ESM-2 variants mapped to their (embedding dimension, final layer).
# Larger variants give slightly better representations at the cost of
# substantially more VRAM and disk space.
ESM2_MODELS = {
    "esm2_t6_8M_UR50D": (320, 6),
    "esm2_t12_35M_UR50D": (480, 12),
    "esm2_t30_150M_UR50D": (640, 30),
    "esm2_t33_650M_UR50D": (1280, 33),
    "esm2_t36_3B_UR50D": (2560, 36),
}

# Default model variant.
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"

# Maximum sequence length accepted by ESM-2. Chains longer than this are
# truncated from the C-terminus with a warning.
ESM2_MAX_SEQ_LEN = 1022



# ==============================================================================
# FILE I/O
# ==============================================================================
# This section ingests the only mandatory input of the pipeline: the single
# DNAproDB JSON file produced by processStructure.py for one complex. No
# separate "models directory" is required — the user is only expected to have
# the working files of the complex being processed (its .pdb, its DNAproDB
# .json, its .pqr, and now its ESM-2 .pt), so the embedding is written right
# next to the JSON it was computed from.


# -----------------------------------------------------------------------------
# Read Single DNAproDB Entry
# -----------------------------------------------------------------------------
def read_entry(json_path: str) -> Dict[str, Any]:
    """
    Load a single-complex DNAproDB JSON file.

    The reader expects exactly one complex per file. A list is also tolerated
    for convenience, in which case only its first element is used. An unreadable
    file is reported and an empty dictionary is returned.

    Args:
        json_path (str): Path to the DNAproDB JSON file of one complex.

    Returns:
        Dict[str, Any]: The parsed complex entry, or an empty dictionary if
            the path is missing or does not hold a readable JSON file.
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] DNAproDB JSON not found at: {json_path}")
        return {}

    if os.path.isdir(json_path):
        print(f"[ERROR] Expected a JSON file, but a directory was found at: {json_path}")
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Tolerate a single-element list, as some exports wrap the entry.
        if isinstance(raw_data, list):
            if not raw_data:
                print(f"[ERROR] Empty JSON list in: {json_path}")
                return {}
            return raw_data[0]

        return raw_data

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to decode JSON {json_path}: {e}")
        return {}
    except Exception as e:
        print(f"[ERROR] Unexpected error reading {json_path}: {e}")
        return {}



# ==============================================================================
# MODEL LOADING
# ==============================================================================
# Loads the ESM-2 protein language model once and exposes the batch converter
# used to tokenize sequences before the forward pass.

# -----------------------------------------------------------------------------
# Load ESM-2 Model
# -----------------------------------------------------------------------------
def load_esm2_model(model_name: str, device: torch.device) -> Tuple[Any, Any]:
    """
    Load an ESM-2 pre-trained model and its batch converter in eval mode.

    The weights are downloaded automatically on first use and cached; later
    runs load from cache. Dropout is disabled so the representations are
    deterministic.

    Args:
        model_name (str): One of the keys in ESM2_MODELS.
        device (torch.device): Hardware device (CPU or CUDA) for inference.

    Returns:
        Tuple[Any, Any]: The ESM-2 model in eval mode and the batch converter
            that turns (label, sequence) pairs into padded tokens.
    """
    print(f"[INFO] Loading ESM-2 model: {model_name}...")
    loader_fn = getattr(esm_module.pretrained, model_name)
    model, alphabet = loader_fn()
    batch_converter = alphabet.get_batch_converter()

    # eval() disables dropout for reproducible, deterministic embeddings.
    model = model.eval().to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print("[INFO] ESM-2 loaded.")
    print(f"Parameters: {num_params:,}.")
    print(f"Device: {device.type.upper()}.")

    return model, batch_converter



# ==============================================================================
# SEQUENCE EXTRACTION
# ==============================================================================
# Pulls the 1-letter sequence and the matching residue id_keys for every protein
# chain of the complex. Sequence order and id_key order must stay perfectly
# aligned, otherwise the embedding of a residue would be mapped to the wrong
# graph node.

# -----------------------------------------------------------------------------
# Extract Protein Sequences and Keys
# -----------------------------------------------------------------------------
def extract_protein_sequences_and_keys(entry: Dict[str, Any]) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Extract the 1-letter sequence and aligned id_keys for each protein chain.

    The id_key is built to match exactly the protein-node id_key produced by
    the graph-construction step, which is the join key used later by the data
    loader. Chains whose sequence length and residue-id count disagree are
    skipped, since the mapping would be misaligned.

    Args:
        entry (Dict[str, Any]): The single complex entry from the raw JSON.

    Returns:
        Dict[str, Tuple[List[str], List[str]]]: Map from chain ID to a tuple of
            (sequence characters, id_keys), both in the same residue order.
    """
    chain_data = {}

    structure_id_key = entry.get("structure_id", "unknown")

    protein_node = entry.get("protein", {})
    chains = protein_node.get("chains", [])

    if not chains:
        print(f"[WARNING] No protein chains found in structure {structure_id_key}")
        return chain_data

    for chain in chains:
        chain_id_key = chain.get("id", "unknown")
        raw_sequence = chain.get("sequence", "unknown")
        raw_residue_id_keys = chain.get("residue_ids", [])

        # Only process valid, non-empty chains.
        if chain_id_key == "unknown" or raw_sequence == "unknown" or not raw_residue_id_keys:
            continue

        # A length mismatch would misalign every downstream embedding, so the
        # whole chain is skipped rather than risk poisoning the dataset.
        if len(raw_sequence) != len(raw_residue_id_keys):
            print(f"[ERROR] Mismatch in {structure_id_key} (chain {chain_id_key}): "
                  f"sequence length ({len(raw_sequence)}) != id count ({len(raw_residue_id_keys)}). Skipping.")
            continue

        seq_chars = list(raw_sequence)

        # Match exactly the node id_key format.
        id_keys = [raw_id_key for raw_id_key in raw_residue_id_keys]

        chain_data[chain_id_key] = (seq_chars, id_keys)

    return chain_data



# ==============================================================================
# EMBEDDING COMPUTATION
# ==============================================================================
# Runs the ESM-2 forward pass per chain and maps each per-residue vector back to
# its spatial id_key, producing the {id_key: tensor} dictionary stored on disk.

# -----------------------------------------------------------------------------
# Extract ESM-2 Embeddings for the Complex
# -----------------------------------------------------------------------------
def extract_embeddings_for_entry(entry: Dict[str, Any], model: Any, batch_converter: Any, device: torch.device, repr_layer: int, use_half: bool) -> Dict[str, torch.Tensor]:
    """
    Compute per-residue ESM-2 embeddings for every protein chain of the complex.

    Chains longer than ESM2_MAX_SEQ_LEN are truncated from the C-terminus and
    their dropped residues receive no embedding.

    Args:
        entry (Dict[str, Any]): The single complex entry from the raw JSON.
        model (Any): Loaded ESM-2 model in eval mode.
        batch_converter (Any): ESM-2 batch converter callable.
        device (torch.device): Inference device.
        repr_layer (int): Transformer layer to read representations from.
        use_half (bool): Store embeddings as float16 to halve disk usage.

    Returns:
        Dict[str, torch.Tensor]: Map from residue id_key to its 1D embedding of
            shape (esm2_dim,).
    """
    embedding_map = {}

    chain_data = extract_protein_sequences_and_keys(entry)

    if not chain_data:
        return embedding_map

    pdb_id_key = entry.get("structure_id", "unknown").lower()

    for chain_id, (seq_chars, id_keys) in chain_data.items():
        # Skip empty chains.
        if not seq_chars or not id_keys:
            continue

        # ESM-2 cannot process sequences longer than ESM2_MAX_SEQ_LEN.
        if len(seq_chars) > ESM2_MAX_SEQ_LEN:
            print(f"[WARNING] Chain {chain_id} of {pdb_id_key.upper()} has {len(seq_chars)} residues (>{ESM2_MAX_SEQ_LEN}).")
            seq_chars = seq_chars[:ESM2_MAX_SEQ_LEN]
            id_keys = id_keys[:ESM2_MAX_SEQ_LEN]

        sequence_str = "".join(seq_chars)
        label = f"{pdb_id_key}_{chain_id}"

        # ------------------------------------------
        # ESM-2 forward pass
        # ------------------------------------------
        _, _, batch_tokens = batch_converter([(label, sequence_str)])
        batch_tokens = batch_tokens.to(device)

        with torch.inference_mode():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

        # Targeted layer, shape [1, seq_len + 2, esm2_dim] (+2 for BOS and EOS).
        token_repr = results["representations"][repr_layer]

        # Drop BOS (position 0) and EOS, keeping residues 1..seq_len.
        residue_repr = token_repr[0, 1: len(seq_chars) + 1, :]

        # Move off the accelerator immediately to free VRAM between chains.
        residue_repr = residue_repr.to("cpu")

        if use_half:
            residue_repr = residue_repr.half()

        # Map each residue vector to its spatial id_key.
        for i, id_key in enumerate(id_keys):
            if id_key != "unknown":
                embedding_map[id_key] = residue_repr[i].clone()

    return embedding_map



# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
# Orchestrates the run for a single complex: read the one DNAproDB JSON entry,
# load the model, embed every protein chain, and write a single .pt file next
# to the input JSON (or to a custom location, if one is given).

# -----------------------------------------------------------------------------
# Main Workflow
# -----------------------------------------------------------------------------
def main(json_path: str, output_path: str = None, model_name: str = ESM2_MODEL_NAME, use_half: bool = False) -> None:
    """
    Compute and persist ESM-2 embeddings for a single complex.

    The protein chains of the complex are embedded and the per-residue vectors
    are saved as a single {id_key: tensor} dict at "output_path", which is
    expected to sit alongside the other working files of the complex being
    processed (its .pdb, its DNAproDB .json, its .pqr, and now its ESM-2 .pt).

    Args:
        json_path (str): Path to the DNAproDB JSON file of the complex.
        output_path (str): Destination .pt file where the embeddings will be
            saved.
        model_name (str): ESM-2 variant key from "ESM2_MODELS".
        use_half (bool): Store embeddings as float16 to halve disk usage.
    """
    if model_name not in ESM2_MODELS:
        print(f"[ERROR] Unknown model '{model_name}'. Valid options: {list(ESM2_MODELS)}")
        return

    esm2_dim, repr_layer = ESM2_MODELS[model_name]

    print("\n# ============================================== #")
    print("#  STARTING ESM-2 EMBEDDING EXTRACTION (SINGLE)  #")
    print("# ============================================== #\n")

    print(f"[INFO] Input JSON path: {json_path}")
    print(f"[INFO] Input ESM-2 model: {model_name} ({esm2_dim} dims, layer {repr_layer}).")
    print(f"[INFO] Storage dtype: {'float16' if use_half else 'float32'}")

    # ------------------------------------------
    # Load the complex
    # ------------------------------------------
    print("\n[1/3] Loading DNAproDB entry...")
    entry = read_entry(json_path)

    if not entry:
        print("[ERROR] No valid data found in JSON. Aborting.")
        return

    pdb_id_key = entry.get("structure_id", "unknown").lower()

    if not pdb_id_key or pdb_id_key == "unknown":
        print("[ERROR] Missing structure_id in the DNAproDB entry. Aborting.")
        return

    print(f"[INFO] Structure identified: {pdb_id_key.upper()}")

    # ------------------------------------------
    # Device selection
    # ------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Hardware accelerator: {device.type.upper()}")

    # ------------------------------------------
    # Load ESM-2 model
    # ------------------------------------------
    print("\n[2/3] Loading ESM-2 model...")
    model, batch_converter = load_esm2_model(model_name, device)

    # ------------------------------------------
    # Embedding extraction
    # ------------------------------------------
    print(f"\n[3/3] Computing embeddings for {pdb_id_key.upper()}...")

    try:
        embedding_map = extract_embeddings_for_entry(
            entry, model, batch_converter, device, repr_layer, use_half
        )

        if not embedding_map:
            print(f"[ERROR] No valid protein embeddings generated for {pdb_id_key.upper()}. Aborting.")
            return

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        torch.save(embedding_map, output_path)

        print(f"[SUCCESS] Embeddings saved to: {output_path}")
        print(f"[INFO] Residues embedded: {len(embedding_map)}")

    except Exception as e:
        print(f"[ERROR] Failed for {pdb_id_key.upper()}: {e}")
        return

    print("\n# ====================================== #")
    print("#  ESM-2 EXTRACTION PIPELINE COMPLETED!  #")
    print("# ====================================== #\n")


# -----------------------------------------------------------------------------
# CLI Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Configure command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Single-complex ESM-2 per-residue embedding extractor from a DNAproDB JSON file.",
        epilog=f"Developed by {__author__}. Version {__version__}.",
    )

    parser.add_argument("-j", "--json", type=str, required=True,
                        help="Path to the DNAproDB JSON file of the complex.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Destination .pt file where the embeddings will be saved.")
    parser.add_argument("--esm", type=str, default=ESM2_MODEL_NAME, choices=list(ESM2_MODELS),
                        help="ESM-2 model variant to use.")
    parser.add_argument("--half", action="store_true",
                        help="Store embeddings as float16 to halve disk usage.")

    args = parser.parse_args()

    main(args.json, args.output, args.esm, args.half)