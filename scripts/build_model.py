# ==============================================================================
# Description: Pipeline to generate the standardized JSON file for GNN training.
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
import glob
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple

import pandas as pd



# ==============================================================================
# VALIDATION & SAFE CASTING HELPERS
# ==============================================================================


# -----------------------------------------------------------------------------
# Safe Float Conversion
# -----------------------------------------------------------------------------
def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Cast a raw value to a float, mapping None or invalid types to a default.

    Args:
        value (Any): The raw value extracted from the JSON dictionary.
        default (float, optional): The fallback value to return if the
            conversion fails. Defaults to 0.0.

    Returns:
        float: The converted numerical value, or the default on failure.
    """
    if value is None or str(value).strip().upper() in ("NA", "NAN", "NONE", ""):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# -----------------------------------------------------------------------------
# Safe Integer Conversion
# -----------------------------------------------------------------------------
def safe_int(value: Any, default: int = 0) -> int:
    """
    Cast a raw value to an integer, mapping None or invalid types to a default.

    Args:
        value (Any): The raw value extracted from the JSON dictionary.
        default (int, optional): The fallback value to return if the
            conversion fails. Defaults to 0.

    Returns:
        int: The converted integer value, or the default on failure.
    """
    if value is None or str(value).strip().upper() in ("NA", "NAN", "NONE", ""):
        return default
    
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# -----------------------------------------------------------------------------
# Safe String Conversion
# -----------------------------------------------------------------------------
def safe_str(value: Any, default: str = "unknown") -> str:
    """
    Cast a raw value to a string, mapping None to a default and cleaning
    invisible whitespace.

    Args:
        value (Any): The raw value extracted from the JSON dictionary.
        default (str, optional): The fallback string to return if the
            value is None. Defaults to "unknown".

    Returns:
        str: The converted string, stripped of leading/trailing whitespace,
             or the default if conversion fails.
    """
    if value is None:
        return default
    
    try:
        # Cast to string and clean up phantom spaces
        result = str(value).strip()
        # Return default if the string ends up completely empty
        return result if result else default
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Safe Boolean Conversion
# -----------------------------------------------------------------------------
def safe_bool(value: Any, default: bool = False) -> bool:
    """
    Evaluate a raw value as a boolean. Safely handles string representations
    like 'false' or '0' that would otherwise incorrectly evaluate to True.

    Args:
        value (Any): The raw value extracted from the JSON dictionary.
        default (bool, optional): The fallback boolean to return if the
            value is None. Defaults to False.

    Returns:
        bool: The evaluated boolean value.
    """
    if value is None:
        return default
        
    if isinstance(value, bool):
        return value
        
    # Catch string representations that Python's bool() gets wrong
    if isinstance(value, str):
        clean_val = value.strip().lower()
        if clean_val in ("false", "0", "no", "f", "n", "none", ""):
            return False
        if clean_val in ("true", "1", "yes", "t", "y"):
            return True
            
    try:
        return bool(value)
    except Exception:
        return default



# ==============================================================================
# DATABASE PARSING & FILE I/O
# ==============================================================================


# -----------------------------------------------------------------------------
# Read DNAproDB Data
# -----------------------------------------------------------------------------
def read_dnaprodb(json_path: str) -> Dict[str, Any]:
    """
    Load DNAproDB JSON data into a flat list of complex entries.

    The reader expects exactly one complex per file. A list is also tolerated
    for convenience, in which case only its first element is used. An
    unreadable file is reported and an empty dictionary is returned, so the
    caller can abort cleanly.

    Args:
        json_path (str): Path to the standardized JSON file of one complex.

    Returns:
        Dict[str, Any]: The parsed complex entry, or an empty dictionary if
            the path is missing or does not hold a readable JSON file.
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] Standardized JSON not found at: {json_path}")
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


# -----------------------------------------------------------------------------
# Read Precomputed Electrostatics
# -----------------------------------------------------------------------------
def read_pqr_charges(pqr_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse precomputed PQR files to extract partial charges and Van der Waals
    radii.

    Args:
        pqr_path (str): Path to the PQR file produced by PDB2PQR.

    Returns:
        Dict[str, Dict[str, Any]]: A single-entry map from the lower-cased
            PDB ID to its {id_key: {"charge": float, "vdw_radius": float}}
            map. The inner map is empty if the file is missing.
    """
    if not os.path.isfile(pqr_path):
        print(f"[ERROR] PQR file not found at: {pqr_path}")
        return {}

    residue_charge_data = {}

    try:
        with open(pqr_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("ATOM"):
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    chain = safe_str(parts[4])
                    number = safe_int(parts[5])

                    charge = safe_float(parts[-2])
                    vdw_radius = safe_float(parts[-1])

                    # Construct the id_key matching the rest of the pipeline.
                    # This assumes that the --chain argument is enabled.
                    id_key = safe_str(f"{chain}.{number}. ")

                    if id_key not in residue_charge_data:
                        residue_charge_data[id_key] = {
                            "charge": 0.0,
                            "vdw_radius": 0.0
                        }

                    residue_charge_data[id_key]["charge"] += charge
                    residue_charge_data[id_key]["vdw_radius"] += vdw_radius

    except Exception as e:
        print(f"[ERROR] Error reading pqr file {pqr_path}: {e}")

    return residue_charge_data



# ==============================================================================
# DATA AGGREGATION PIPELINE
# ==============================================================================
# This section extracts, processes, and integrates the parsed databases into a
# unified, high-density JSON architecture optimized for GNN training. Each
# extract_* function reads one logical block from a DNAproDB entry and returns a
# normalized structure; the builder class then assembles those blocks into the
# final node/edge/global-context layout.
#
# Convention: only the first model (models[0]) of every block is used, since the
# target structures are single-model X-ray/cryo-EM complexes.

# -----------------------------------------------------------------------------
# Extract Residue Nomenclature
# -----------------------------------------------------------------------------
def extract_residue_info(entry: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract basic nomenclature (names and chemical names) for nucleotides and
    amino acids.

    Nucleotides and amino acids are kept in two separate lists so that a protein
    chain and a DNA chain sharing the same letter and residue number can never
    overwrite each other.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Two lists, one for
            nucleotides and one for amino acids, each holding the nomenclature
            dictionary of every residue in JSON order.
    """
    dna_info_list = []
    protein_info_list = []

    # ------------------------------------------
    # Nucleotide nomenclature
    # ------------------------------------------
    dna_node = entry.get("dna", {})
    nucleotides = dna_node.get("nucleotides", [])

    if not nucleotides:
        print(f"[WARNING] No nucleotides found in structure {entry.get('structure_id', 'unknown')}")

    for nucleotide in nucleotides: 
        dna_info = {
            "id_key": safe_str(nucleotide.get("id", "unknown")),
            "name": safe_str(nucleotide.get("name", "unknown")),
            "chemical_name": safe_str(nucleotide.get("chemical_name", "unknown")),
            "name_short": safe_str(nucleotide.get("name_short", "unknown")),
            "chain": safe_str(nucleotide.get("chain", "unknown")),
            "number": safe_int(nucleotide.get("number", 0)),
        }

        dna_info_list.append(dna_info)


    # ------------------------------------------
    # Amino acid nomenclature
    # ------------------------------------------
    protein_node = entry.get("protein", {})
    aminoacids = protein_node.get("residues", [])

    if not aminoacids:
        print(f"[WARNING] No amino acids found in structure {entry.get('structure_id', 'unknown')}")

    for aminoacid in aminoacids: 
        protein_info = {
            "id_key": safe_str(aminoacid.get("id", "unknown")),
            "name": safe_str(aminoacid.get("name", "unknown")),
            "chemical_name": safe_str(aminoacid.get("chemical_name", "unknown")),
            "name_short": safe_str(aminoacid.get("name_short", "unknown")),
            "chain": safe_str(aminoacid.get("chain", "unknown")),
            "number": safe_int(aminoacid.get("number", 0)),
        }

        protein_info_list.append(protein_info)

    return dna_info_list, protein_info_list


# -----------------------------------------------------------------------------
# Extract DNA Shape Parameters
# -----------------------------------------------------------------------------
def extract_dna_shape_parameters(entry: Dict[str, Any]) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[Tuple[str, str], Dict[str, float]]]:
    """
    Extract the per-step DNA shape parameters from the first DNA model.

    For every helical segment the intra-base-pair parameters (buckle, propeller,
    opening, etc.) are keyed by both orderings of a base pair, and the
    inter-base-pair-step parameters (twist, roll, rise, etc.) are keyed by both
    orderings of consecutive steps along each strand. These maps are later joined
    onto the base-pair and stacking edges.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.

    Returns:
        Tuple[Dict[Tuple[str, str], Dict[str, float]],
              Dict[Tuple[str, str], Dict[str, float]]]: The intra-pair and
            inter-step shape-parameter maps, each keyed by an (id_key, id_key)
            tuple.
    """
    intra_shape_parameters = {}
    inter_shape_parameters = {}

    # Intra-pair parameters live on the base pair; inter-step parameters on the
    # dinucleotide step between consecutive base pairs.
    intra_keys = ["buckle", "propeller", "opening", "shear", "stagger", "stretch"]
    inter_keys = ["twist", "roll", "rise", "slide", "tilt", "shift"]

    dna_node = entry.get("dna", {})
    models = dna_node.get("models", [])

    if not models:
        print(f"[WARNING] No DNA models found in structure {entry.get('structure_id', 'unknown')}")
        return intra_shape_parameters, inter_shape_parameters

    entities = models[0].get("entities", [])

    for entity in entities:
        helical_segments = entity.get("helical_segments", [])

        for segment in helical_segments:
            shape_parameters = segment.get("shape_parameters", {})

            ids1 = [safe_str(id_key) for id_key in segment.get("ids1", [])]
            ids2 = [safe_str(id_key) for id_key in segment.get("ids2", [])]

            # ------------------------------------------
            # Intra-pair parameters
            # ------------------------------------------
            for i, (id1, id2) in enumerate(zip(ids1, ids2)):
                pair_values = {}

                for key in intra_keys:
                    array = shape_parameters.get(key, [])
                    pair_values[key] = safe_float(array[i]) if i < len(array) else 0.0

                intra_shape_parameters[(id1, id2)] = pair_values
                intra_shape_parameters[(id2, id1)] = pair_values

            # ------------------------------------------
            # Inter-pair parameters
            # ------------------------------------------
            for i in range(len(ids1) - 1):
                stack_values = {}

                for key in inter_keys:
                    array = shape_parameters.get(key, [])
                    stack_values[key] = safe_float(array[i]) if i < len(array) else 0.0

                inter_shape_parameters[(ids1[i], ids1[i + 1])] = stack_values
                inter_shape_parameters[(ids1[i + 1], ids1[i])] = stack_values
                inter_shape_parameters[(ids2[i], ids2[i + 1])] = stack_values
                inter_shape_parameters[(ids2[i + 1], ids2[i])] = stack_values

    return intra_shape_parameters, inter_shape_parameters


# -----------------------------------------------------------------------------
# Extract Residue Interface Data
# -----------------------------------------------------------------------------
def extract_residue_interface_data(entry: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Extract the residue interface summary from the first interface model.

    DNAproDB stores, for every amino acid that contacts the DNA, its helicoidal
    coordinates (phi, s, rho: the cylindrical placement relative to the DNA
    helical axis), the buried accessible surface area it contributes, and how
    many nucleotides it touches.

    Also it reports how many amino acids each nucleotide contacts and the buried
    accessible surface area it contributes, so the nucleotide nodes can expose
    an interaction_count and an interface BASA instead of the empty placeholder.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]: A map from
            amino acid ID key to its interface summary, and a map from nucleotide
            ID key to its interface summary. Residues away from the interface are
            simply absent.
    """
    aminoacid_interface_data = {}
    nucleotide_interface_data = {}

    interfaces_node = entry.get("interfaces", {})
    models = interfaces_node.get("models", [])

    if not models:
        print(f"[WARNING] No interfaces found in structure {entry.get('structure_id', 'unknown')}")
        return aminoacid_interface_data, nucleotide_interface_data

    for interface in models[0]:
        aminoacids = interface.get("residue_data", [])
        nucleotides = interface.get("nucleotide_data", [])

        for aminoacid in aminoacids:
            # ------------------------------------------
            # Identity
            # ------------------------------------------
            id_key = safe_str(aminoacid.get("res_id", "unknown"))
            
            # ------------------------------------------
            # Helicoidal coordinates
            # ------------------------------------------
            helicoidal_coordinates = aminoacid.get("helicoidal_coordinates", {})

            aminoacid_interface_data[id_key] = {
                "interaction_count": safe_int(aminoacid.get("nucleotide_interaction_count", 0)),
                "helicoidal_coordinates": {
                    "phi": safe_float(helicoidal_coordinates.get("phi", 0.0)),
                    "s": safe_float(helicoidal_coordinates.get("s", 0.0)),
                    "rho": safe_float(helicoidal_coordinates.get("rho", 0.0)),
                },
            }

        for nucleotide in nucleotides:
            # ------------------------------------------
            # Identity
            # ------------------------------------------
            id_key =  safe_str(nucleotide.get("nuc_id", "unknown"))

            nucleotide_interface_data[id_key] = {
                "interaction_count": safe_int(nucleotide.get("residue_interaction_count", 0)),
            }

    return aminoacid_interface_data, nucleotide_interface_data


# -----------------------------------------------------------------------------
# Extract Nucleotide Nodes
# -----------------------------------------------------------------------------
def extract_nucleotides(entry: Dict[str, Any], nucleotide_interface_data: Dict[str, Dict[str, Any]],
                        partial_charge_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build the nucleotide node blocks of a complex.

    Every nucleotide contributes its conformational state, the fractional
    accessible surface area (FASA) by moiety, the interface interaction count and
    the aggregated PQR partial charge. Nucleotides away from the interface, or
    belonging to a complex with no precomputed electrostatics, keep zeros in the
    corresponding fields instead of being dropped, so the node list always
    mirrors dna.nucleotides one to one.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.
        nucleotide_interface_data (Dict[str, Dict[str, Any]]): Map from
            nucleotide ID key to its interface summary, from
            extract_residue_interface_data.
        partial_charge_data (Dict[str, Dict[str, Any]]): Map from ID key to its
            {"charge": X, "vdw_radius": Y} block.

    Returns:
        List[Dict[str, Any]]: One dictionary per nucleotide, in JSON order.
    """
    nucleotide_list = []

    dna_node = entry.get("dna", {})
    nucleotides = dna_node.get("nucleotides", [])

    if not nucleotides:
        print(f"[WARNING] No nucleotides found in structure {entry.get('structure_id', 'unknown')}")
        return nucleotide_list

    for nucleotide in nucleotides:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        id_key = safe_str(nucleotide.get("id", "unknown"))

        # ------------------------------------------
        # Structure
        # ------------------------------------------
        # Conformational descriptors are stored as single-element lists.
        secondary_structure = nucleotide.get("secondary_structure", [])
        glycosidic_conformation = nucleotide.get("glycosidic_conformation", [])

        # ------------------------------------------
        # Surface metrics
        # ------------------------------------------
        # Fractional Accessible Surface Area (FASA).
        fasa = nucleotide.get("fasa", [])
        fasa_data = fasa[0] if fasa else {}

        # ------------------------------------------
        # Electrostatics
        # ------------------------------------------
        # A nucleotide with no precomputed PDB2PQR/APBS run (e.g. the run
        # failed or has not been generated yet for this complex) keeps zeros.
        nucleotide_partial_charge_data = partial_charge_data.get(id_key, {})

        # ------------------------------------------
        # Interface data
        # ------------------------------------------
        # A nucleotide away from the interface keeps zeros.
        interface_data = nucleotide_interface_data.get(id_key, {})

        nucleotide_data = {
            "id_key": id_key,
            "structure": {
                "secondary_structure": safe_str(secondary_structure[0]) if secondary_structure else "unknown",
                "glycosidic_conformation": safe_str(glycosidic_conformation[0]) if glycosidic_conformation else "unknown",
                "phosphate_present": safe_bool(nucleotide.get("phosphate_present", False)),
                "modified": safe_bool(nucleotide.get("modified", False)),
            },
            "surface_metrics": {
                "fasa": {
                    "bs": safe_float(fasa_data.get("bs", 0.0)),
                    "pp": safe_float(fasa_data.get("pp", 0.0)),
                    "sg": safe_float(fasa_data.get("sg", 0.0)),
                    "sr": safe_float(fasa_data.get("sr", 0.0)),
                    "wg": safe_float(fasa_data.get("wg", 0.0)),
                },
            },
            "interaction_metrics": {
                "interaction_count": safe_int(interface_data.get("interaction_count", 0)),
            },
            "electrostatics": {
                "partial_charge": safe_float(nucleotide_partial_charge_data.get("charge", 0.0)),
            },
        }

        nucleotide_list.append(nucleotide_data)

    return nucleotide_list


# -----------------------------------------------------------------------------
# Extract Amino Acid Nodes
# -----------------------------------------------------------------------------
def extract_aminoacids(entry: Dict[str, Any], aminoacid_interface_data: Dict[str, Dict[str, Any]],
                       partial_charge_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build the amino acid node blocks of a complex.

    Every amino acid contributes its secondary structure, the helicoidal
    coordinates of its contact with the DNA, the surface descriptors (circular
    variance, SAP score, FASA and SESA split into main chain and side chain), the
    interface interaction count and the aggregated PQR partial charge. Amino
    acids away from the interface, or belonging to a complex with no precomputed
    electrostatics, keep zeros in the corresponding fields instead of being
    dropped, so the node list always mirrors protein.residues one to one.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.
        aminoacid_interface_data (Dict[str, Dict[str, Any]]): Map from amino acid
            ID key to its interface summary, from extract_residue_interface_data.
        partial_charge_data (Dict[str, Dict[str, Any]]): Map from ID key to its
            {"charge": X, "vdw_radius": Y} block.

    Returns:
        List[Dict[str, Any]]: One dictionary per amino acid, in JSON order.
    """
    aminoacid_list = []

    protein_node = entry.get("protein", {})
    aminoacids = protein_node.get("residues", [])

    if not aminoacids:
        print(f"[WARNING] No aminoacids found in structure {entry.get('structure_id', 'unknown')}")
        return aminoacid_list

    for aminoacid in aminoacids:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        id_key = safe_str(aminoacid.get("id", "unknown"))

        # ------------------------------------------
        # Structure
        # ------------------------------------------
        secondary_structure = aminoacid.get("secondary_structure", [])

        # Fractional Accessible Surface Area (FASA).
        fasa = aminoacid.get("fasa", [])
        fasa_data = fasa[0] if fasa else {}

        # Solvent excluded surface area (SESA).
        sesa = aminoacid.get("sesa", [])
        sesa_data = sesa[0] if sesa else {}

        # ------------------------------------------
        # Surface metrics
        # ------------------------------------------
        sap_score = aminoacid.get("sap_score", [])
        cv_coarse = aminoacid.get("cv_coarse", [])
        cv_fine = aminoacid.get("cv_fine", [])

        # ------------------------------------------
        # Electrostatics summary
        # ------------------------------------------
        # An amino acid with no precomputed PDB2PQR/APBS run for this complex
        # keeps zeros.
        aminoacid_partial_charge_data = partial_charge_data.get(id_key, {})

        # ------------------------------------------
        # Interface data
        # ------------------------------------------
        # An amino acid away from the interface keeps zeros.
        interface_data = aminoacid_interface_data.get(id_key, {})
        interface_helicoidal = interface_data.get("helicoidal_coordinates", {})

        aminoacid_data = {
            "id_key": id_key,
            "structure": {
                "secondary_structure": safe_str(secondary_structure[0]) if secondary_structure else "unknown",
                "helicoidal_coordinates": {
                    "phi": safe_float(interface_helicoidal.get("phi", 0.0)),
                    "s": safe_float(interface_helicoidal.get("s", 0.0)),
                    "rho": safe_float(interface_helicoidal.get("rho", 0.0)),
                },
            },
            "surface_metrics": {
                "cv_coarse": safe_float(cv_coarse[0]) if cv_coarse else 0.0,
                "cv_fine": safe_float(cv_fine[0]) if cv_fine else 0.0,
                "sap_score": safe_float(sap_score[0]) if sap_score else 0.0,
                "fasa": {
                    "mc": safe_float(fasa_data.get("mc", 0.0)),
                    "sc": safe_float(fasa_data.get("sc", 0.0)),
                },
                "sesa": {
                    "mc": safe_float(sesa_data.get("mc", 0.0)),
                    "sc": safe_float(sesa_data.get("sc", 0.0)),
                },
            },
            "interaction_metrics": {
                "interaction_count": safe_int(interface_data.get("interaction_count", 0)),
            },
            "electrostatics": {
                "partial_charge": safe_float(aminoacid_partial_charge_data.get("charge", 0.0)),
            },
        }

        aminoacid_list.append(aminoacid_data)

    return aminoacid_list


# -----------------------------------------------------------------------------
# Extract DNA Base-Stacking Edges
# -----------------------------------------------------------------------------
def extract_dna_stacks(entry: Dict[str, Any], inter_shape_parameters: Dict[Tuple[str, str], Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Extract DNA base-stacking interactions from the first DNA model.

    Each stack links two nucleotides and reports the minimum inter-atomic
    distance between the stacked bases, plus the inter-base-pair-step shape
    parameters attached from inter_shape_parameters.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.
        inter_shape_parameters (Dict[Tuple[str, str], Dict[str, float]]):
            Inter-base-pair-step shape parameters keyed by (id_key, id_key),
            from extract_dna_shape_parameters.

    Returns:
        List[Dict[str, Any]]: One dictionary per stacking interaction.
    """
    stacks_list = []

    dna_node = entry.get("dna", {})
    models = dna_node.get("models", [])

    if not models:
        print(f"[WARNING] No DNA models found in structure {entry.get('structure_id', 'unknown')}")
        return stacks_list

    stacks = models[0].get("stacks", [])

    for stack in stacks:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        source_id_key = safe_str(stack.get("id1", "unknown"))
        target_id_key = safe_str(stack.get("id2", "unknown"))

        # ------------------------------------------
        # Shape parameters
        # ------------------------------------------
        shape_parameters = inter_shape_parameters.get((source_id_key, target_id_key), {})

        stack_data = {
            "source_id_key": source_id_key,
            "target_id_key": target_id_key,
            "geometry": {
                "mindist": safe_float(stack.get("mindist", 0.0)),
            },
            "shape_parameters": {
                "twist": safe_float(shape_parameters.get("twist", 0.0)),
                "roll": safe_float(shape_parameters.get("roll", 0.0)),
                "rise": safe_float(shape_parameters.get("rise", 0.0)),
                "slide": safe_float(shape_parameters.get("slide", 0.0)),
                "tilt": safe_float(shape_parameters.get("tilt", 0.0)),
                "shift": safe_float(shape_parameters.get("shift", 0.0)),
            },
        }

        stacks_list.append(stack_data)

    return stacks_list


# -----------------------------------------------------------------------------
# Extract DNA Backbone Edges
# -----------------------------------------------------------------------------
def extract_dna_links(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract DNA backbone links (phosphodiester bonds) from the first DNA model.

    Each link is directed 3' -> 5' and reports the bonded atoms on both ends and
    the bond distance.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.

    Returns:
        List[Dict[str, Any]]: One dictionary per backbone link.
    """
    link_list = []

    dna_node = entry.get("dna", {})
    models = dna_node.get("models", [])

    if not models:
        print(f"[WARNING] No DNA models found in structure {entry.get('structure_id', 'unknown')}")
        return link_list

    links = models[0].get("links", [])

    for link in links:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        source_id_key = safe_str(link.get("3p_nuc_id", "unknown"))
        target_id_key = safe_str(link.get("5p_nuc_id", "unknown"))

        link_data = {
            "source_id_key": source_id_key,
            "target_id_key": target_id_key,
            "geometry": {
                "link_distance": safe_float(link.get("link_distance", 0.0)),
            },
        }

        link_list.append(link_data)

    return link_list


# -----------------------------------------------------------------------------
# Extract DNA Base-Pairing Edges
# -----------------------------------------------------------------------------
def extract_dna_pairs(entry: Dict[str, Any], intra_shape_parameters: Dict[Tuple[str, str], Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Extract DNA base pairs and their hydrogen bonds from the first DNA model.

    Each pair reports its coarse pairing type, a mismatch flag, the number of
    hydrogen bonds that hold the bases together, and the intra-base-pair shape
    parameters attached from intra_shape_parameters.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.
        intra_shape_parameters (Dict[Tuple[str, str], Dict[str, float]]):
            Intra-base-pair shape parameters keyed by (id_key, id_key), from
            extract_dna_shape_parameters.

    Returns:
        List[Dict[str, Any]]: One dictionary per base pair.
    """
    pair_list = []

    dna_node = entry.get("dna", {})
    models = dna_node.get("models", [])

    if not models:
        print(f"[WARNING] No DNA models found in structure {entry.get('structure_id', 'unknown')}")
        return pair_list

    pairs = models[0].get("pairs", [])

    for pair in pairs:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        source_id_key = safe_str(pair.get("id1", "unknown"))
        target_id_key = safe_str(pair.get("id2", "unknown"))

        # ------------------------------------------
        # Hydrogen bonds
        # ------------------------------------------
        hbonds = pair.get("hbonds", [])
        hbond_distances = []

        for hbond in hbonds:
            hbond_distances.append(safe_float(hbond.get("distance", 0.0)))

        # ------------------------------------------
        # Shape parameters
        # ------------------------------------------
        shape_parameters = intra_shape_parameters.get((source_id_key, target_id_key), {})

        pair_data = {
            "source_id_key": source_id_key,
            "target_id_key": target_id_key,
            "structure": {
                "pair_type": safe_str(pair.get("pair_type", "unknown")),
                "mismatched": safe_bool(pair.get("mismatched", False)),
            },
            "chemistry": {
                "hbonds": {
                    "total": len(hbonds),
                    "distances": hbond_distances,
                }
                
            },
            "shape_parameters": {
                "buckle": safe_float(shape_parameters.get("buckle", 0.0)),
                "propeller": safe_float(shape_parameters.get("propeller", 0.0)),
                "opening": safe_float(shape_parameters.get("opening", 0.0)),
                "shear": safe_float(shape_parameters.get("shear", 0.0)),
                "stagger": safe_float(shape_parameters.get("stagger", 0.0)),
                "stretch": safe_float(shape_parameters.get("stretch", 0.0)),
            },
        }

        pair_list.append(pair_data)

    return pair_list


# -----------------------------------------------------------------------------
# Extract Protein Backbone Edges
# -----------------------------------------------------------------------------
def extract_protein_backbone(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract protein backbone (peptide-bond) edges from the first protein model.

    Consecutive residue IDs inside every segment are linked pairwise. DNAproDB
    segments can reference residues that never make it into protein.residues, so
    both endpoints are checked against the residue table and any dangling pair is
    dropped. The gap attribute is reserved for chain breaks and is currently
    always zero.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.

    Returns:
        List[Dict[str, Any]]: One dictionary per peptide-bond edge.
    """
    backbone_list = []

    protein_node = entry.get("protein", {})
    aminoacids = protein_node.get("residues", [])

    if not aminoacids:
        print(f"[WARNING] No aminoacids found in structure {entry.get('structure_id', 'unknown')}")
        return backbone_list

    valid_ids = {safe_str(aminoacid.get("id", "unknown")) for aminoacid in aminoacids}

    models = protein_node.get("models", [])

    if not models:
        print(f"[WARNING] No models found in structure {entry.get('structure_id', 'unknown')}")
        return backbone_list
    
    segments = models[0].get("segments", [])

    if not segments:
        print(f"[WARNING] No protein segments found in structure {entry.get('structure_id', 'unknown')}")
        return backbone_list

    for segment in segments:
        id_keys = [safe_str(id_key) for id_key in segment.get("residue_ids", [])]

        for source_id_key, target_id_key in zip(id_keys[:-1], id_keys[1:]):

            if source_id_key not in valid_ids or target_id_key not in valid_ids:
                continue

            link_data = {
                "source_id_key": source_id_key,
                "target_id_key": target_id_key,
                "geometry": {
                    "gap": 0
                },
            }
                
            backbone_list.append(link_data)
            
    return backbone_list


# -----------------------------------------------------------------------------
# Extract Protein-DNA Interaction Edges
# -----------------------------------------------------------------------------
def extract_protein_dna_interactions(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract bipartite protein-DNA interaction edges from the interface model.

    Each interaction maps a contacting amino acid-nucleotide pair and gathers
    extensive contact metrics: spatial geometry, surface areas (BASA) and
    chemistry (hydrogen-bond and VdW counts by moiety, plus the individual bond
    distances, with the hydrogen bonds split into direct and water-mediated).

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.

    Returns:
        List[Dict[str, Any]]: One dictionary per protein-DNA contact edge.
    """
    interaction_list = []

    interfaces_node = entry.get("interfaces", {})
    models = interfaces_node.get("models", [])

    # For interfaces, models[0] is directly a list of interface dictionaries.
    if not models:
        print(f"[WARNING] No interaction models found in structure {entry.get('structure_id', 'unknown')}")
        return interaction_list

    for interface in models[0]:
        interactions = interface.get("nucleotide-residue_interactions", [])

        for interaction in interactions:
            # ------------------------------------------
            # Identity
            # ------------------------------------------
            aminoacid_id_key = safe_str(interaction.get("res_id", "unknown"))
            nucleotide_id_key = safe_str(interaction.get("nuc_id", "unknown"))
            
            # ------------------------------------------
            # Surface metrics
            # ------------------------------------------
            # Buried Accessible Surface Area (BASA).
            basa = interaction.get("basa", {})

            basa_bs = basa.get("bs", {})
            basa_pp = basa.get("pp", {})
            basa_sg = basa.get("sg", {})
            basa_sr = basa.get("sr", {})
            basa_wg = basa.get("wg", {})

            # ------------------------------------------
            # Chemistry
            # ------------------------------------------
            # Hydrogen bonds.
            hbond_sum = interaction.get("hbond_sum", {})

            hbond_sum_bs = hbond_sum.get("bs", {})
            hbond_sum_pp = hbond_sum.get("pp", {})
            hbond_sum_sg = hbond_sum.get("sg", {})
            hbond_sum_sr = hbond_sum.get("sr", {})
            hbond_sum_wg = hbond_sum.get("wg", {})

            hbonds = interaction.get("hbonds", [])
            hbond_direct_distances = []
            hbond_water_mediated_distances = []

            for hbond in hbonds:
                # A direct bond carries the literal string "NA" in water_id. safe_str only
                # maps None to its default, not "NA", so the sentinel is tested explicitly.
                water_id = safe_str(hbond.get("water_id"), "NA")

                if water_id.upper() in ("NA", "NAN", "NONE", ""):
                    hbond_direct_distances.append(safe_float(hbond.get("distance", 0.0)))
                else:
                    hbond_water_mediated_distances.append(safe_float(hbond.get("distance_WA", 0.0)))

            # Van der Waals interactions.
            vdw_sum = interaction.get("vdw_sum", {})

            vdw_sum_bs = vdw_sum.get("bs", {})
            vdw_sum_pp = vdw_sum.get("pp", {})
            vdw_sum_sg = vdw_sum.get("sg", {})
            vdw_sum_sr = vdw_sum.get("sr", {})
            vdw_sum_wg = vdw_sum.get("wg", {})

            vdw_interactions = interaction.get("vdw_interactions", [])
            vdw_distances = []
    
            for vdw_interaction in vdw_interactions:
                vdw_distances.append(safe_float(vdw_interaction.get("distance", 0.0)))

            interaction_data = {
                "aminoacid_id_key": aminoacid_id_key,
                "nucleotide_id_key": nucleotide_id_key,
                "geometry": {
                    "min_distance": safe_float(interaction.get("min_distance", 0.0)),
                    "mean_nn_distance": safe_float(interaction.get("mean_nn_distance", 0.0)),
                    "cm_distance": safe_float(interaction.get("cm_distance", 0.0)),
                    "geometry_type": safe_str(interaction.get("geometry", "unknown")),
                },
                "surface_metrics": {
                    "basa": {
                        "bs": {
                            "mc": safe_float(basa_bs.get("mc", 0.0)),
                            "sc": safe_float(basa_bs.get("sc", 0.0)),
                        },
                        "pp": {
                            "mc": safe_float(basa_pp.get("mc", 0.0)),
                            "sc": safe_float(basa_pp.get("sc", 0.0)),
                        },
                        "sg": {
                            "mc": safe_float(basa_sg.get("mc", 0.0)),
                            "sc": safe_float(basa_sg.get("sc", 0.0)),
                        },
                        "sr": {
                            "mc": safe_float(basa_sr.get("mc", 0.0)),
                            "sc": safe_float(basa_sr.get("sc", 0.0)),
                        },
                        "wg": {
                            "mc": safe_float(basa_wg.get("mc", 0.0)),
                            "sc": safe_float(basa_wg.get("sc", 0.0)),
                        },
                    },
                },
                "chemistry": {
                    "is_weak_interaction": safe_bool(interaction.get("weak_interaction", False)),
                    "hbonds": {
                        "total": {
                            "bs": {
                                "mc": safe_int(hbond_sum_bs.get("mc", 0)),
                                "sc": safe_int(hbond_sum_bs.get("sc", 0)),
                            },
                            "pp": {
                                "mc": safe_int(hbond_sum_pp.get("mc", 0)),
                                "sc": safe_int(hbond_sum_pp.get("sc", 0)),
                            },
                            "sg": {
                                "mc": safe_int(hbond_sum_sg.get("mc", 0)),
                                "sc": safe_int(hbond_sum_sg.get("sc", 0)),
                            },
                            "sr": {
                                "mc": safe_int(hbond_sum_sr.get("mc", 0)),
                                "sc": safe_int(hbond_sum_sr.get("sc", 0)),
                            },
                            "wg": {
                                "mc": safe_int(hbond_sum_wg.get("mc", 0)),
                                "sc": safe_int(hbond_sum_wg.get("sc", 0)),
                            },
                        },
                        "direct_distances": hbond_direct_distances,
                        "water_mediated_distances": hbond_water_mediated_distances,
                    },
                    "vdw_interactions": {
                        "total": {
                            "bs": {
                                "mc": safe_int(vdw_sum_bs.get("mc", 0)),
                                "sc": safe_int(vdw_sum_bs.get("sc", 0)),
                            },
                            "pp": {
                                "mc": safe_int(vdw_sum_pp.get("mc", 0)),
                                "sc": safe_int(vdw_sum_pp.get("sc", 0)),
                            },
                            "sg": {
                                "mc": safe_int(vdw_sum_sg.get("mc", 0)),
                                "sc": safe_int(vdw_sum_sg.get("sc", 0)),
                            },
                            "sr": {
                                "mc": safe_int(vdw_sum_sr.get("mc", 0)),
                                "sc": safe_int(vdw_sum_sr.get("sc", 0)),
                            },
                            "wg": {
                                "mc": safe_int(vdw_sum_wg.get("mc", 0)),
                                "sc": safe_int(vdw_sum_wg.get("sc", 0)),
                            },
                        },
                        "distances": vdw_distances,
                    },
                },
            }

            interaction_list.append(interaction_data)

    return interaction_list


# -----------------------------------------------------------------------------
# Extract Interfaces Global Context
# -----------------------------------------------------------------------------
def extract_interfaces(entry: Dict[str, Any], resolution, temperature, ph, dna_descriptor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the protein-chain interface descriptors from the first interface
    model.

    DNAproDB decomposes a single protein-DNA interface into one feature block
    per protein chain (so a homo-tetramer bound to one duplex yields four
    blocks). Each block reports buried surface area, hydrogen-bond and Van der
    Waals counts (all broken down by groove / sugar / phosphate moiety and by
    secondary structure), amino acid propensities, mean hydrophobicity, surface
    geometry and the pseudo-pair / pseudo-stack ratios.

    These blocks are deliberately kept UNPOOLED: collapsing them (e.g. a plain
    mean) blurs heterogeneous chains, so aggregate_feature_dicts only transposes
    them field by field and every leaf becomes a list with one value per protein
    chain. The data loader then decides how to aggregate them (extensive
    quantities summed, intensive ones weighted by the interaction count,
    categorical ones by weighted majority, plus a learned set / attention
    read-out for the GNN).

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.
        conditions (Dict[str, Dict[str, Any]]): Map from lower-cased PDB ID to
            its conditions dictionary, extracted from the original PDB files.
        dna_descriptor_data (Dict[str, Any]): The complex-level DNA descriptors,
            from extract_dna_global_descriptors, carried through untouched.

    Returns:
        Dict[str, Any]: The global-context block: the interface count, the
            experimental conditions, the per-chain interface features and the DNA
            descriptors. Counts drop to zero and the interface map is empty if no
            interface model is present.
    """
    feature_list = []
    
    interfaces_node = entry.get("interfaces", {})
    models = interfaces_node.get("models", [])

    if not models:
        print(f"[WARNING] No interfaces found in structure {entry.get('structure_id', 'unknown')}.")
        return {
            "num_interfaces": 0,
            "resolution": safe_float(resolution, 2.5),
            "temperature": safe_float(temperature, 298.0),
            "pH": safe_float(ph, 7.4),
            "interfaces": {},
            "dna_descriptors": dna_descriptor_data,
        }

    for interface in models[0]:
        features = interface.get("interface_features", [])

        for feature in features:
            # ------------------------------------------
            # Protein surface geometry
            # ------------------------------------------
            surface_geometry = feature.get("protein_surface_geometry", {})

            cv_coarse = surface_geometry.get("cv_coarse", {})
            cv_fine = surface_geometry.get("cv_fine", {})

            # ------------------------------------------
            # Propensities
            # ------------------------------------------
            aminoacid_propensities = feature.get("residue_propensities", {})

            feature_data = {       
                "interaction_metrics": {
                    "hydrophobicity": safe_float(feature.get("mean_hydrophobicity_score", 0.0)),
                    "num_interactions": safe_int(feature.get("interaction_count", 0)),
                    "num_weak_interactions": safe_int(feature.get("weak_interaction_count", 0)),
                },
                "protein_surface_geometry": {
                    "cv_coarse": {
                        "flat_ratio": safe_float(cv_coarse.get("flat_ratio", 0.0)),
                        "valley_ratio": safe_float(cv_coarse.get("valley_ratio", 0.0)),
                        "peak_ratio": safe_float(cv_coarse.get("peak_ratio", 0.0)),
                    },
                    "cv_fine": {
                        "flat_ratio": safe_float(cv_fine.get("flat_ratio", 0.0)),
                        "valley_ratio": safe_float(cv_fine.get("valley_ratio", 0.0)),
                        "peak_ratio": safe_float(cv_fine.get("peak_ratio", 0.0)),
                    },
                },
                "aminoacid_propensities": {
                    "ALA": safe_float(aminoacid_propensities.get("ALA", 0.0)),
                    "ARG": safe_float(aminoacid_propensities.get("ARG", 0.0)),
                    "ASN": safe_float(aminoacid_propensities.get("ASN", 0.0)),
                    "ASP": safe_float(aminoacid_propensities.get("ASP", 0.0)),
                    "CYS": safe_float(aminoacid_propensities.get("CYS", 0.0)),
                    "GLN": safe_float(aminoacid_propensities.get("GLN", 0.0)),
                    "GLU": safe_float(aminoacid_propensities.get("GLU", 0.0)),
                    "GLY": safe_float(aminoacid_propensities.get("GLY", 0.0)),
                    "HIS": safe_float(aminoacid_propensities.get("HIS", 0.0)),
                    "ILE": safe_float(aminoacid_propensities.get("ILE", 0.0)),
                    "LEU": safe_float(aminoacid_propensities.get("LEU", 0.0)),
                    "LYS": safe_float(aminoacid_propensities.get("LYS", 0.0)),
                    "MET": safe_float(aminoacid_propensities.get("MET", 0.0)),
                    "PHE": safe_float(aminoacid_propensities.get("PHE", 0.0)),
                    "PRO": safe_float(aminoacid_propensities.get("PRO", 0.0)),
                    "SER": safe_float(aminoacid_propensities.get("SER", 0.0)),
                    "THR": safe_float(aminoacid_propensities.get("THR", 0.0)),
                    "TRP": safe_float(aminoacid_propensities.get("TRP", 0.0)),
                    "TYR": safe_float(aminoacid_propensities.get("TYR", 0.0)),
                    "VAL": safe_float(aminoacid_propensities.get("VAL", 0.0)),
                },
            }
        
            feature_list.append(feature_data)

    global_context = {
        "num_interfaces": len(feature_list),
        "resolution": safe_float(resolution, 2.5),
        "temperature": safe_float(temperature, 298.0),
        "pH": safe_float(ph, 7.4),
        "interfaces": aggregate_feature_dicts(feature_list),
        "dna_descriptors": dna_descriptor_data,
    }

    return global_context


# -----------------------------------------------------------------------------
# Extract DNA Global Descriptors
# -----------------------------------------------------------------------------
def extract_dna_global_descriptors(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the complex-level DNA descriptors from the first DNA model.

    Reports one block per DNA entity (topology and chemical modifications) and
    one per helical segment (classification, sequence composition, helical-axis
    geometry and structural content). Both families are transposed by
    aggregate_feature_dicts, so every leaf becomes a list with one value per
    entity or per segment and the data loader keeps full control of the pooling.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.

    Returns:
        Dict[str, Any]: The entity and helical-segment descriptor maps plus their
            counts. Counts drop to zero and both maps are empty if no DNA model
            is present.
    """
    entity_list = []
    helical_segment_list = []

    dna_node = entry.get("dna", {})
    models = dna_node.get("models", [])

    if not models:
        print(f"[WARNING] No DNA models found in structure {entry.get('structure_id', 'unknown')}")
        return {
            "num_entities": 0,
            "num_helical_segments": 0,
            "entities": {},
            "helical_segments": {}
        }

    entities = models[0].get("entities", [])

    for entity in entities:
        modifications = entity.get("chemical_modifications", {})

        entity_data = {
            "topology": {
                "entity_type": safe_str(entity.get("type", "other")),
                "num_strands": safe_int(entity.get("num_strands", 0)),
                "num_helical_segments": safe_int(entity.get("num_helical_segments", 0)),
                "num_single_stranded_segments": safe_int(entity.get("num_single-stranded_segments", 0)),
            },
            "modifications": {
                "methylated_cytosine": safe_bool(modifications.get("5_methylated_cytosine", False)),
                "non_standard_nucleotides": safe_bool(modifications.get("non-standard_nucleotides", False)),
            },
        }
        
        entity_list.append(entity_data)
        
        helical_segments = entity.get("helical_segments", [])

        for segment in helical_segments:
            helical_axis = segment.get("helical_axis", {})

            helical_segment_data = {
                "classification": safe_str(segment.get("classification", "other")),
                "sequence": {
                    "length": safe_int(segment.get("length", 0)),
                    "GC_content": safe_float(segment.get("GC_content")),
                },
                "geometry": {
                    "mean_radius": safe_float(segment.get("mean_radius")),
                    "axis_curvature": safe_str(helical_axis.get("axis_curvature", "other")),
                    "axis_length": safe_float(helical_axis.get("axis_length", 0.0)),
                },
                "content": {
                    "a_tracts": safe_bool(segment.get("contains_A-tracts", False)),
                    "hoogsteen_pairs": safe_bool(segment.get("contains_hoogsteen_pairs", False)),
                    "mismatches": safe_bool(segment.get("contains_mismatches", False)),
                    "non_wc_pairs": safe_bool(segment.get("contains_non-wc_pairs", False)),
                },
            }

            helical_segment_list.append(helical_segment_data)

    dna_descriptor_data = {
        "num_entities": len(entity_list),
        "num_helical_segments": len(helical_segment_list),
        "entities": aggregate_feature_dicts(entity_list),
        "helical_segments": aggregate_feature_dicts(helical_segment_list),
    }

    return dna_descriptor_data


# -----------------------------------------------------------------------------
# Aggregate Feature Dictionaries
# -----------------------------------------------------------------------------
def aggregate_feature_dicts(feature_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Transpose a list of identically-structured feature dictionaries into a
    single dictionary that preserves the nested layout but turns every leaf
    value into a list collecting that field across all input dictionaries.

    Example:
        [{"a": 1, "b": {"c": 2}}, {"a": 3, "b": {"c": 4}}]
        -> {"a": [1, 3], "b": {"c": [2, 4]}}

    All dictionaries are assumed to share the same schema (guaranteed here
    because every per-chain feature block is built by the same code). An empty
    input yields an empty dictionary.

    Args:
        feature_dicts (List[Dict[str, Any]]): One feature dictionary per protein
            chain, all with the same nested structure.

    Returns:
        Dict[str, Any]: A dictionary mirroring the input structure, with each
            leaf replaced by the list of per-chain values.
    """
    if not feature_dicts:
        return {}

    # ---------------------------------------------------------------------
    # Recursive Merge Helper
    # ---------------------------------------------------------------------
    def merge(values: List[Any]) -> Any:
        """
        Recursively merge a list of values based on their type.

        Args:
            values (List[Any]): A list of values (either nested dictionaries
                or scalar features) extracted across multiple chains.

        Returns:
            Any: A dictionary with aggregated lists as leaves, or a flat
                list if the input values were scalars.
        """
        first = values[0]

        # Recurse key by key using the first element as the schema.
        if isinstance(first, dict):
            return {key: merge([value[key] for value in values]) for key in first}

        # Collect the per-chain values into a list.
        return list(values)

    return merge(feature_dicts)


# -----------------------------------------------------------------------------
# Extract Protein Structural Classes
# -----------------------------------------------------------------------------
def extract_protein_class(entry: Dict[str, Any]) -> List[str]:
    """
    Extract the CATH architecture classes for the protein chains in a complex.

    DNAproDB stores the CATH hierarchy at the complex level under
    'meta_data.cath', providing lists of structural classifications per level
    ("Class", "Architecture", "Topology", "Homology"). This function isolates 
    the "Architecture" annotations.

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.

    Returns:
        List[str]: A list of CATH architecture names present in the complex.
    """
    classes = []
    
    metadata = entry.get("meta_data", {})
    cath = metadata.get("cath", {})
    architectures = cath.get("Architecture", [])

    for architecture in architectures:
        classes.append(safe_str(architecture))
    
    if not classes:
        return ["unknown"]
    
    return classes


# -----------------------------------------------------------------------------
# Extract DNA Structural Classes
# -----------------------------------------------------------------------------
def extract_dna_class(entry: Dict[str, Any]) -> str:
    """
    Categorize the gross structural state of the DNA based on the number of strands.

    Evaluates the 'num_chains' attribute inside the DNA node to classify the 
    nucleic acid as single-stranded (ssDNA), double-stranded (dsDNA), or 
    multi-stranded (e.g., triplexes, G-quadruplexes).

    Args:
        entry (Dict[str, Any]): A single complex entry from the raw JSON.

    Returns:
        str: The structural class ("ssDNA", "dsDNA", or "multi_strand").
    """
    dna = entry.get("dna", {})
    models = dna.get("models", [])

    if not models:
        print(f"[WARNING] No DNA models found in structure {entry.get('structure_id', 'unknown')}")
        return "unknown"
    
    entities = models[0].get("entities", [])

    num_strands = sum(safe_int(entity.get("num_strands", 0)) for entity in entities)
    num_helical = sum(safe_int(entity.get("num_helical_segments", 0)) for entity in entities)
    num_single = sum(safe_int(entity.get("num_single-stranded_segments", 0)) for entity in entities)


    if num_helical == 0:
        return "ssDNA"
    if num_strands == 2 and num_single == 0:
        return "dsDNA"

    return "multi_strand"



# ==============================================================================
# STANDARDIZED JSON BUILDER
# ==============================================================================
# Implements the Builder pattern to assemble the final per-complex JSON from the
# parsed blocks. The build_* methods populate the nodes, edges, global-context,
# and target sections in turn and return self to allow method chaining.

# -----------------------------------------------------------------------------
# Standardized JSON Builder Class
# -----------------------------------------------------------------------------
class StandardizedJSONBuilder:
    """
    Assemble the final standardized JSON for a single complex.

    All raw blocks are extracted once at construction time; the build_* methods
    then format them into the nodes / edges / global_context / target layout
    expected by the GNN data loader.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, entry: Dict[str, Any], electrostatics: Dict[str, Dict[str, Any]], resolution: float, temperature: float, ph: float) -> None:
        """
        Pre-extract every structural, evolutionary, and metadata block needed to
        build the final model.

        Args:
            entry (Dict[str, Any]): A single complex entry from the raw JSON.
            electrostatics (Dict[str, Dict[str, Any]]): Map from lower-cased PDB
                ID to its {id_key: {"charge": X, "vdw_radius": Y}} map.
            resolution (float): Experimental resolution of the structure.
            temperature (float): Experimental temperature, in Kelvin.
            ph (float): Experimental pH of the structure.
        """
        # Metadata.
        self.dna_info, self.protein_info = extract_residue_info(entry)

        # Nodes.
        self.aminoacid_interface_data, self.nucleotide_interface_data = extract_residue_interface_data(entry)
        self.nucleotide_nodes = extract_nucleotides(entry, self.nucleotide_interface_data, electrostatics)
        self.aminoacid_nodes = extract_aminoacids(entry, self.aminoacid_interface_data, electrostatics)

        # Edges.
        self.intra_shape_parameters, self.inter_shape_parameters = extract_dna_shape_parameters(entry)
        self.dna_links = extract_dna_links(entry)
        self.dna_stacks = extract_dna_stacks(entry, self.inter_shape_parameters)
        self.dna_pairs = extract_dna_pairs(entry, self.intra_shape_parameters)

        self.protein_dna_interactions = extract_protein_dna_interactions(entry)

        # Protein backbone (peptide-bond) edges. DNAproDB's segment residue_ids
        # can reference a few more residues per chain than actually end up as
        # nodes in protein.residues (observed on real entries), so any edge
        # whose endpoint has no matching node is dropped to avoid a dangling
        # reference in the final graph.
        valid_aminoacid_ids = {node["id_key"] for node in self.aminoacid_nodes}
        self.protein_backbone = [
            edge for edge in extract_protein_backbone(entry)
            if edge["source_id_key"] in valid_aminoacid_ids and edge["target_id_key"] in valid_aminoacid_ids
        ]

        # Global context.
        self.dna_descriptors = extract_dna_global_descriptors(entry)
        self.interfaces = extract_interfaces(entry, resolution, temperature, ph, self.dna_descriptors)
        self.protein_class = extract_protein_class(entry)
        self.dna_class = extract_dna_class(entry)

        # Raw entry kept for reference.
        self.entry = entry

        # Skeleton of the final standardized model.
        self.model = {
            "pdb_id": safe_str(entry.get("structure_id", "unknown")).lower(),
            "nodes": {},
            "edges": {},
            "global_context": {},
            "classes": {},
            "nomenclatures": {},
        }

    # -------------------------------------------------------------------------
    # Build Nodes Section
    # -------------------------------------------------------------------------
    def build_nodes(self) -> "StandardizedJSONBuilder":
        """
        Assemble the nodes section from the pre-extracted node lists.

        The per-node enrichment (structural state, surface metrics,
        physico-chemical properties, and interface summary) is performed by 
        the extract_* functions; this method only places the nucleotide and 
        residue lists under the "dna" and "protein" keys.

        Returns:
            StandardizedJSONBuilder: The current instance, for method chaining.
        """
        self.model["nodes"] = {
            "dna": self.nucleotide_nodes,
            "protein": self.aminoacid_nodes,
        }

        return self

    # -------------------------------------------------------------------------
    # Build Edges Section
    # -------------------------------------------------------------------------
    def build_edges(self) -> "StandardizedJSONBuilder":
        """
        Build the edge section from the four edge types.

        Produces backbone links, base stacks, base pairs (intra-DNA), and
        protein-DNA interactions (bipartite). Edges are stored in a single
        direction; the data loader is expected to add the reverse edges for the
        undirected DNA edge types. Node attributes are referenced by ID only to
        avoid duplicating the per-node feature vectors stored in the nodes
        section.

        Returns:
            StandardizedJSONBuilder: The current instance, for method chaining.
        """
        self.model["edges"] = {
            "dna_base_pairs": self.dna_pairs,
            "dna_stacks": self.dna_stacks,
            "dna_backbone": self.dna_links,
            "protein_backbone": self.protein_backbone,
            "protein_dna_interactions": self.protein_dna_interactions,
        }

        return self

    # -------------------------------------------------------------------------
    # Build Global Context Section
    # -------------------------------------------------------------------------
    def build_global_context(self) -> "StandardizedJSONBuilder":
        """
        Assemble the global-context section.

        DNAproDB decomposes the interface into one feature block per protein
        chain; these unpooled blocks are stored as a list so the data loader can
        decide how to aggregate them (see extract_interfaces).

        Returns:
            StandardizedJSONBuilder: The current instance, for method chaining.
        """
        self.model["global_context"] = self.interfaces

        return self
    
    # -------------------------------------------------------------------------
    # Build Classes Section
    # -------------------------------------------------------------------------
    def build_classes(self) -> "StandardizedJSONBuilder":
        """
        Inject the structural classifications into the model.

        This section stores the CATH architecture composition for the protein 
        chains and the strandedness category (ssDNA, dsDNA, etc.) for the DNA.

        Returns:
            StandardizedJSONBuilder: The current instance, for method chaining.
        """
        self.model["classes"] = {
            "protein": self.protein_class,
            "dna": self.dna_class,
        }

        return self

    # -------------------------------------------------------------------------
    # Build Nomenclature Section
    # -------------------------------------------------------------------------
    def build_nomenclatures(self) -> "StandardizedJSONBuilder":
        """
        Build the nomenclature section mapping IDs to standard and chemical names.

        This separates the bulky string metadata from the numeric node features,
        keeping the graph input matrices clean while preserving human-readable
        labels.

        Returns:
            StandardizedJSONBuilder: The current instance, for method chaining.
        """
        self.model["nomenclatures"] = {
            "dna": self.dna_info,
            "protein": self.protein_info,
        }

        return self

    # -------------------------------------------------------------------------
    # Get Final Result
    # -------------------------------------------------------------------------
    def get_result(self) -> Dict[str, Any]:
        """
        Return the fully assembled standardized model.

        Returns:
            Dict[str, Any]: The final dictionary with all structural,
                            evolutionary, and thermodynamic data.
        """
        return self.model



# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
# Entry point that orchestrates the end-to-end workflow: command-line parsing,
# loading the source databases, building one standardized JSON per matched
# complex, and writing the results to disk.

# -----------------------------------------------------------------------------
# Main Workflow
# -----------------------------------------------------------------------------
def main(json_path: str, pqr_path: str, resolution: float, temperature: float, ph: float, output_path: str) -> None:
    """
    Build the standardized JSON of a single complex.

    Combines the raw DNAproDB entry with the electrostatics computed by
    PDB2PQR and the experimental conditions entered by the user.

    Args:
        json_path (str): Path to the raw DNAproDB JSON file of the complex.
        pqr_path (str): Path to the PQR file of the complex, from PDB2PQR.
        resolution (float): Experimental resolution of the structure.
        temperature (float): Experimental temperature, in Kelvin.
        ph (float): Experimental pH of the structure.
        output_path (str): Destination path for the standardized JSON.
    """
    print("\n# ================================================ #")
    print("#  STARTING STANDARDIZED JSON GENERATION PIPELINE  #")
    print("# ================================================ #\n")

    print(f"[INFO] Input JSON path: {json_path}")
    print(f"[INFO] Input PQR path: {pqr_path}")
    print(f"[INFO] Conditions: resolution={resolution}, temperature={temperature}, pH={ph}")
    print(f"[INFO] Output path: {output_path}")

    # ------------------------------------------
    # Load the complex
    # ------------------------------------------
    print("\n[1/3] Loading DNAproDB entry...")
    entry = read_dnaprodb(json_path)

    if not entry:
        print("[ERROR] No valid data found in JSON. Aborting.")
        return

    pdb_id_key = safe_str(entry.get("structure_id", "unknown")).lower()

    if not pdb_id_key or pdb_id_key == "unknown":
        print("[ERROR] Missing structure_id in the DNAproDB entry. Aborting.")
        return

    print(f"[INFO] Structure identified: {pdb_id_key.upper()}")

    # ------------------------------------------
    # Load electrostatics and conditions
    # ------------------------------------------
    print("\n[2/3] Loading electrostatics and conditions...")
    charges = read_pqr_charges(pqr_path)

    # ------------------------------------------
    # Build and export the standardized JSON
    # ------------------------------------------
    print(f"\n[3/3] Building the standardized JSON for {pdb_id_key.upper()}...")

    try:
        # No known affinity for a new complex being predicted.
        builder = StandardizedJSONBuilder(entry, charges, resolution, temperature, ph)
        builder.build_nodes() \
            .build_edges() \
            .build_global_context() \
            .build_classes() \
            .build_nomenclatures()

        model = builder.get_result()

        if not model:
            print(f"[ERROR] Construction failed for {pdb_id_key.upper()}. Aborting.")
            return

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(model, out_file, indent=4)

        print(f"[SUCCESS] Standardized JSON saved to: {output_path}")

    except Exception as e:
        print(f"[ERROR] Unexpected error in {pdb_id_key.upper()}: {e}")
        return

    print("\n# ========================================= #")
    print("#  STANDARDIZED JSON GENERATION COMPLETED!  #")
    print("# ========================================= #\n")


# -----------------------------------------------------------------------------
# CLI Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Configure command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Pipeline to generate a standardized JSON from PDB and raw JSON data.",
        epilog=f"Developed by {__author__}. Version {__version__}.",
    )

    parser.add_argument("-j", "--json", type=str, required=True,
                        help="Path to the raw DNAproDB JSON file of the complex.")
    parser.add_argument("-p", "--pqr", type=str, required=True,
                        help="Path to the PQR file of the complex, from PDB2PQR.")
    parser.add_argument("--resolution", type=float, required=True,
                        help="Experimental resolution of the structure.")
    parser.add_argument("--temperature", type=float, required=True,
                        help="Experimental temperature of the structure, in Kelvin.")
    parser.add_argument("--ph", type=float, required=True,
                        help="Experimental pH of the structure.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Destination path for the standardized JSON.")

    args = parser.parse_args()

    main(args.json, args.pqr, args.resolution, args.temperature, args.ph, args.output)
