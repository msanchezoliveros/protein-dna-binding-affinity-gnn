# ==============================================================================
# Description: Pipeline to predict the binding affinity of one protein-DNA
#              complex from its standardized JSON file and its ESM-2 embeddings,
#              using the trained cross-validation ensemble.
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
import copy
import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentionalAggregation, to_hetero, global_add_pool, GATv2Conv
from torch_geometric.utils import softmax as pyg_softmax



# ==============================================================================
# ENCODING VOCABULARIES
# ==============================================================================
# Category vocabularies for the one-hot encoders and the edge-attribute
# dimensions.

# ------------------------------------------
# Node categorical vocabularies
# ------------------------------------------
STANDARD_AMINOACID_MAP = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]

STANDARD_NUCLEOTIDE_MAP = [
    "DA", "DC", "DG", "DT", "DU", "DI"
]

AMINOACID_SECONDARY_STRUCTURE_MAP = [
    "H", "S", "L"
]

NUCLEOTIDE_SECONDARY_STRUCTURE_MAP = [
    "helical", "other", "single-stranded"
]

NUCLEOTIDE_GLYCOSIDIC_CONFORMATION_MAP = [
    "anti", "syn"
]

# ------------------------------------------
# Edge categorical vocabularies
# ------------------------------------------
BASE_PAIR_TYPE_MAP = [
    "watson-crick", "other", "hoogsteen"
]

INTERACTION_GEOMETRY_TYPE_MAP = [
    "none", "pseudo_pair", "pseudo_stack"
]

# ------------------------------------------
# Global context categorical vocabularies
# ------------------------------------------
DNA_CLASS_MAP = [
    "dsDNA", "ssDNA", "multi_strand"
]

DNA_ENTITY_TYPE_MAP = [
    "perfect_helix", "imperfect_helix", "irregular_helix", "ssDNA", "other"
]

DNA_HELICAL_SEGMENT_CLASSIFICATION_MAP = [
    "A-DNA", "B-DNA", "Z-DNA", "other"
]

DNA_HELICAL_SEGMENT_AXIS_CURVATURE_MAP = [
    "in-plane", "linear", "non-planar"
]



# ==============================================================================
# DEFAULT HYPERPARAMETERS
# ==============================================================================
# Baseline configuration used whenever cross_validate is called.

HYPERPARAMETERS = {
    # ------------------------------------------
    # Architecture & Dimensionality
    # ------------------------------------------
    "node_hidden_dim": 32,
    "node_heads": 4,
    "edge_dim": 32,
    "edge_hidden_multiplier": 2,
    "esm_projected_dim": 32,
    "esm_hidden_multiplier": 4,
    "global_projected_dim": 8,
    "global_hidden_multiplier": 4,
    "head_hidden_dim": 24,

    # ------------------------------------------
    # Regularisation
    # ------------------------------------------
    "node_proj_dropout": 0.5,
    "node_conv_dropout": 0.5,
    "edge_dropout": 0.5,
    "esm_dropout": 0.5,
    "global_dropout": 0.5,
    "head_dropout": 0.5,

    # ------------------------------------------
    # Training
    # ------------------------------------------
    "batch_size": 128,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "huber_delta": 0.5,

    # ------------------------------------------
    # Flow control
    # ------------------------------------------
    "scheduler_patience": 5,
    "early_stopping_patience": 200
}



# ==============================================================================
# FEATURE DIMENSIONS
# ==============================================================================
# Expected length of every feature vector, together with the column indexes that
# hold continuous values.

# ------------------------------------------
# Node index values
# ------------------------------------------
DNA_NODE_DIM = 20
DNA_NODE_NORM_INDEXES = list(range(13, 20))

PROTEIN_NODE_DIM = 35
PROTEIN_NODE_NORM_INDEXES = list(range(23, 35))

# ------------------------------------------
# Edge index values
# ------------------------------------------
DNA_BASE_PAIR_EDGES_DIM = 13
DNA_BASE_PAIR_EDGES_NORM_INDEXES = list(range(4, 13))

DNA_STACK_EDGES_DIM = 7
DNA_STACK_EDGES_NORM_INDEXES = list(range(0, 7))

DNA_BACKBONE_EDGES_DIM = 1
DNA_BACKBONE_EDGES_NORM_INDEXES = [0]

PROTEIN_BACKBONE_EDGES_DIM = 1
PROTEIN_BACKBONE_EDGES_NORM_INDEXES = [0]

PROTEIN_DNA_INTERACTION_EDGES_DIM = 46
PROTEIN_DNA_INTERACTION_EDGES_NORM_INDEXES = (list(range(0, 3)) + list(range(6, 16)) + list(range(17, 46)))

# ------------------------------------------
# Global context index values
# ------------------------------------------
GLOBAL_CONTEXT_DIM = 96
GLOBAL_CONTEXT_NORM_INDEXES = list(range(63)) + list(range(68, 74)) + list(range(76, 81)) + [85, 86, 90, 91]

# ------------------------------------------
# ESM-2 index values
# ------------------------------------------
ESM_DIM = 1280



# ==============================================================================
# ONE-HOT ENCODER
# ==============================================================================
# Vocabulary one-hot encoder used for every categorical node and edge
# descriptor. Unknown categories map to an all-zero vector.

# -----------------------------------------------------------------------------
# One-Hot Encoder Class
# -----------------------------------------------------------------------------
class OneHotEncoder:
    """
    Encode a value against a fixed vocabulary into a one-hot float vector.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, vocabulary: List[str]) -> None:
        """
        Store the vocabulary and pre-compute its category to column mapping.

        Args:
            vocabulary (List[str]): The ordered list of valid categories.
        """
        self.vocabulary = list(vocabulary)
        self.vector_size = len(self.vocabulary)
        self.mapping = {
            category: index for index, category in enumerate(self.vocabulary)
        }

    # -------------------------------------------------------------------------
    # Encode Value
    # -------------------------------------------------------------------------
    def encode(self, value: str) -> List[float]:
        """
        Encode a single value into its one-hot vector.

        Composite labels are split on "/" and every recognised part switches its
        own column on, so a residue annotated as "H/S" degrades to a multi-hot
        vector instead of losing both states.

        Args:
            value (str): The category to encode.

        Returns:
            List[float]: A one-hot vector, all zeros if the value is unknown.
        """
        encoded_vector = [0.0] * self.vector_size

        for part in value.split("/"):
            index = self.mapping.get(part, -1)
            if index != -1:
                encoded_vector[index] = 1.0

        return encoded_vector

    # -------------------------------------------------------------------------
    # Vector Size
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        """
        Returns:
            int: The number of columns this encoder contributes.
        """
        return self.vector_size


# -----------------------------------------------------------------------------
# Generate Encoders
# -----------------------------------------------------------------------------
def generate_encoders() -> Dict[str, OneHotEncoder]:
    """
    Build one OneHotEncoder per categorical vocabulary.

    Returns:
        Dict[str, OneHotEncoder]: A map from vocabulary name to its encoder.
    """
    encoder_vocabularies = {
        "STANDARD_AMINOACID_MAP": STANDARD_AMINOACID_MAP,
        "STANDARD_NUCLEOTIDE_MAP": STANDARD_NUCLEOTIDE_MAP,
        "AMINOACID_SECONDARY_STRUCTURE_MAP": AMINOACID_SECONDARY_STRUCTURE_MAP,
        "NUCLEOTIDE_SECONDARY_STRUCTURE_MAP": NUCLEOTIDE_SECONDARY_STRUCTURE_MAP,
        "NUCLEOTIDE_GLYCOSIDIC_CONFORMATION_MAP": NUCLEOTIDE_GLYCOSIDIC_CONFORMATION_MAP,
        "BASE_PAIR_TYPE_MAP": BASE_PAIR_TYPE_MAP,
        "INTERACTION_GEOMETRY_TYPE_MAP": INTERACTION_GEOMETRY_TYPE_MAP,
        "DNA_CLASS_MAP": DNA_CLASS_MAP,
        "DNA_ENTITY_TYPE_MAP": DNA_ENTITY_TYPE_MAP,
        "DNA_HELICAL_SEGMENT_CLASSIFICATION_MAP": DNA_HELICAL_SEGMENT_CLASSIFICATION_MAP,
        "DNA_HELICAL_SEGMENT_AXIS_CURVATURE_MAP": DNA_HELICAL_SEGMENT_AXIS_CURVATURE_MAP,
    }

    return {
        name: OneHotEncoder(vocabulary) for name, vocabulary in encoder_vocabularies.items()
    }



# ==============================================================================
# LIST REDUCERS
# ==============================================================================
# Every list-valued field in the standardized model is reduced to a fixed number
# of columns here. The reducer is chosen by the physical nature of the quantity.

# -----------------------------------------------------------------------------
# Extensive Quantities Statistical Panel
# -----------------------------------------------------------------------------
def reduce_extensive(values: List[float]) -> List[float]:
    """
    Reduce an extensive quantity (areas, counts) to its total and its mean.

    The sum carries the size of the complex and the mean the typical per-chain
    (or per-segment) contribution, so both the absolute magnitude and the
    normalized one stay available to the model.

    Args:
        values (List[float]): The per-chain / per-segment values.

    Returns:
        List[float]: The [sum, mean] pair, or two zeros if the list is empty.
    """
    array = np.asarray(values if values is not None else [], dtype=float)

    if array.size == 0:
        return [0.0, 0.0]

    return [
        float(np.sum(array)), float(np.mean(array))
    ]


# -----------------------------------------------------------------------------
# Intensive Quantities Statistical Panel
# -----------------------------------------------------------------------------
def reduce_intensive(values: List[float], weights: List[float]) -> List[float]:
    """
    Reduce an intensive quantity (ratios, scores) to its weighted mean and its
    dispersion.

    Summing an intensive quantity is meaningless, so the chains are averaged
    weighting each one by its own contribution (interaction count or strand
    length). The standard deviation is kept as a heterogeneity signal: it is what
    tells apart a symmetric homodimer from a complex where one chain does most of
    the binding.

    Args:
        values (List[float]): The per-chain / per-segment values.
        weights (List[float]): The weight of every value. Ignored if its length
            does not match the values or if it sums to zero, in which case a
            plain mean is used.

    Returns:
        List[float]: The [weighted mean, standard deviation] pair, or two zeros
            if the list is empty.
    """
    array = np.asarray(values if values is not None else [], dtype=float)

    if array.size == 0:
        return [0.0, 0.0]

    weight_array = np.asarray(weights if weights is not None else [], dtype=float)

    if weight_array.size == array.size and weight_array.sum() > 0:
        mean = float(np.average(array, weights=weight_array))
    else:
        mean = float(np.mean(array))

    # A single-element list has zero dispersion by definition, not missing data.
    return [
        mean, float(np.std(array))
    ]


# -----------------------------------------------------------------------------
# Flag Reducer
# -----------------------------------------------------------------------------
def reduce_flag(values: List[Any]) -> List[float]:
    """
    Collapse a list of boolean flags with an OR.

    A modification present in a single entity or segment already describes the
    complex, so the presence is what matters and not how many blocks carry it.

    Args:
        values (List[Any]): The per-chain / per-segment flags.

    Returns:
        List[float]: A single-element list holding 1.0 if any flag is set.
    """
    if not values:
        return [0.0]

    return [
        1.0 if any(bool(value) for value in values) else 0.0
    ]


# -----------------------------------------------------------------------------
# Categorical Reducer
# -----------------------------------------------------------------------------
def reduce_categorical(values: List[str], encoder: OneHotEncoder, weights: List[float]) -> List[float]:
    """
    Collapse a list of categories into the one-hot of its weighted majority.

    Every label accumulates its own weight and the dominant one is encoded, so a
    long B-DNA duplex is not outvoted by a two-base-pair irregular segment.

    Args:
        values (List[str]): The per-chain / per-segment labels.
        encoder (OneHotEncoder): The encoder of the corresponding vocabulary.
        weights (List[float]): The weight of every label. Missing, null or zero
            weights fall back to 1.0, degrading to a plain vote count.

    Returns:
        List[float]: The one-hot vector of the winning label, all zeros if the
            list is empty.
    """
    if not values:
        return [0.0] * len(encoder)

    totals = {}
    for index, label in enumerate(values):
        weight = 1.0
        if weights is not None and index < len(weights):
            weight = float(weights[index]) or 1.0
        key = str(label)
        totals[key] = totals.get(key, 0.0) + weight

    return encoder.encode(max(totals, key=totals.get))


# -----------------------------------------------------------------------------
# Bond Distance Reducer
# -----------------------------------------------------------------------------
def reduce_distances(distances: List[float]) -> List[float]:
    """
    Reduce a variable-length list of bond distances to a fixed panel.

    The count keeps the strength of the contact (how many bonds hold the pair
    together), the minimum its tightest geometry and the mean its overall
    quality. Non-positive distances are treated as missing measurements and are
    excluded from the statistics, but they still count towards the total.

    Args:
        distances (List[float]): The individual bond distances of one edge.

    Returns:
        List[float]: The [count, minimum, mean] panel, with zeroed statistics if
            no valid distance is present.
    """
    valid = [float(distance) for distance in (distances or []) if float(distance) > 0.0]

    if not valid:
        return [float(len(distances or [])), 0.0, 0.0]

    return [
        float(len(distances or [])), float(min(valid)), float(sum(valid) / len(valid))
    ]



# ==============================================================================
# FILESYSTEM HELPERS
# ==============================================================================
# Directory-listing helpers shared across the pipeline. Hidden and temporary
# entries are always ignored.

# -----------------------------------------------------------------------------
# Filter Hidden and Temporary Entries
# -----------------------------------------------------------------------------
def is_hidden_or_temp(name: str) -> bool:
    """
    Whether a base name belongs to a hidden (".") or temporary ("~") entry.

    Args:
        name (str): The file or directory base name to evaluate.

    Returns:
        bool: True if the name starts with "." or "~", False otherwise.
    """
    return name.startswith(".") or name.startswith("~")


# -----------------------------------------------------------------------------
# List Regular Files
# -----------------------------------------------------------------------------
def list_files(directory: str, pattern: str = "*") -> List[str]:
    """
    List the regular files in a directory matching a glob pattern.

    Hidden and temporary entries are skipped, and sub-directories are never
    returned.

    Args:
        directory (str): Directory to scan.
        pattern (str): Glob pattern applied inside the directory (e.g. "*.pdb").

    Returns:
        List[str]: Paths of the matching files.
    """
    return [
        path for path in glob.glob(os.path.join(directory, pattern))
        if os.path.isfile(path) and not is_hidden_or_temp(os.path.basename(path))
    ]



# ==============================================================================
# NODE EXTRACTION
# ==============================================================================
# Build the node index maps and the per-node feature vectors for DNA nucleotides
# and protein amino acids.

# -----------------------------------------------------------------------------
# DNA Node Index and Nomenclature
# -----------------------------------------------------------------------------
def dna_node_info(entry: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]:
    """
    Build the DNA node index and nomenclature maps.

    The index map is derived from the nomenclature list itself ensuring it
    perfectly aligns with the feature matrix built by dna_node_features and the
    edge index references.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.

    Returns:
        Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]: The {id_key: row index}
            map and the {id_key: nomenclature} map.
    """
    index_map = {}
    info_map = {}

    nomenclature_data = {}

    nodes = entry.get("nodes", {})
    dna_nodes = nodes.get("dna", [])

    nomenclatures = entry.get("nomenclatures", {})
    dna_nomenclature = nomenclatures.get("dna", [])

    for nomenclature in dna_nomenclature:
        id_key = str(nomenclature.get("id_key", "unknown"))

        nomenclature_data[id_key] = {
            "name": str(nomenclature.get("name", "UNK")),
            "chemical_name": str(nomenclature.get("chemical_name", "unknown")),
            "name_short": str(nomenclature.get("name_short", "unknown")),
            "chain": str(nomenclature.get("chain", "unknown")),
            "number": int(nomenclature.get("number", 0)),
        }

    # ------------------------------------------
    # Map Construction
    # ------------------------------------------
    for index, node in enumerate(dna_nodes):
        id_key = str(node.get("id_key", "unknown"))
        index_map[id_key] = index

        nomenclature = nomenclature_data.get(id_key, {})

        info_map[id_key] = {
            "name": str(nomenclature.get("name", "UNK")),
            "chemical_name": str(nomenclature.get("chemical_name", "unknown")),
            "name_short": str(nomenclature.get("name_short", "unknown")),
            "chain": str(nomenclature.get("chain", "unknown")),
            "number": int(nomenclature.get("number", 0)),
        }

    return index_map, info_map


# -----------------------------------------------------------------------------
# Protein Node Index and Nomenclature
# -----------------------------------------------------------------------------
def protein_node_info(entry: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]:
    """
    Build the protein node index and nomenclature maps.

    The index map is derived from the nomenclature list itself ensuring it
    perfectly aligns with the feature matrix built by protein_node_features and
    the edge index references.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.

    Returns:
        Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]: The {id_key: row index}
            map and the {id_key: nomenclature dictionary} map.
    """
    index_map = {}
    info_map = {}

    nomenclature_data = {}

    nodes = entry.get("nodes", {})
    protein_nodes = nodes.get("protein", [])

    nomenclatures = entry.get("nomenclatures", {})
    protein_nomenclature = nomenclatures.get("protein", [])

    for nomenclature in protein_nomenclature:
        id_key = str(nomenclature.get("id_key", "unknown"))

        nomenclature_data[id_key] = {
            "name": str(nomenclature.get("name", "UNK")),
            "chemical_name": str(nomenclature.get("chemical_name", "unknown")),
            "name_short": str(nomenclature.get("name_short", "unknown")),
            "chain": str(nomenclature.get("chain", "unknown")),
            "number": int(nomenclature.get("number", 0)),
        }

    # ------------------------------------------
    # Map Construction
    # ------------------------------------------
    for index, node in enumerate(protein_nodes):
        id_key = str(node.get("id_key", "unknown"))
        index_map[id_key] = index

        nomenclature = nomenclature_data.get(id_key, {})

        info_map[id_key] = {
            "name": str(nomenclature.get("name", "UNK")),
            "chemical_name": str(nomenclature.get("chemical_name", "unknown")),
            "name_short": str(nomenclature.get("name_short", "unknown")),
            "chain": str(nomenclature.get("chain", "unknown")),
            "number": int(nomenclature.get("number", 0)),
        }

    return index_map, info_map


# -----------------------------------------------------------------------------
# DNA Node Features
# -----------------------------------------------------------------------------
def dna_node_features(entry: Dict[str, Any], encoder: Dict[str, OneHotEncoder],
                      dna_data_map: Dict[str, Dict[str, Any]]) -> List[List[float]]:
    """
    Build the per-nucleotide feature vectors.

    Each vector concatenates the one-hot nucleotide identity, the one-hot
    secondary structure and glycosidic conformation, the structural flags, the
    fractional accessible surface area (FASA) by moiety, the interface
    interaction count and the aggregated partial charge.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.
        encoder (Dict[str, OneHotEncoder]): The categorical encoders.
        dna_data_map (Dict[str, Dict[str, Any]]): Nucleotide nomenclature keyed
            by ID, used for the name one-hot.

    Returns:
        List[List[float]]: One feature vector per nucleotide, in node order.
    """
    nodes_list = []

    nodes = entry.get("nodes", {})
    dna_nodes = nodes.get("dna", [])

    for node in dna_nodes:
        nucleotide_node = []

        id_key = str(node.get("id_key", "unknown"))
        nucleotide_info = dna_data_map.get(id_key, {})

        # ------------------------------------------
        # Identity
        # ------------------------------------------
        name = str(nucleotide_info.get("name", "UNK"))
        nucleotide_node += encoder["STANDARD_NUCLEOTIDE_MAP"].encode(name)

        # ------------------------------------------
        # Structure
        # ------------------------------------------
        structure = node.get("structure", {})

        secondary_structure = str(structure.get("secondary_structure", "unknown"))
        nucleotide_node += encoder["NUCLEOTIDE_SECONDARY_STRUCTURE_MAP"].encode(secondary_structure)

        glycosidic_conformation = str(structure.get("glycosidic_conformation", "unknown"))
        nucleotide_node += encoder["NUCLEOTIDE_GLYCOSIDIC_CONFORMATION_MAP"].encode(glycosidic_conformation)

        nucleotide_node.append(float(structure.get("phosphate_present", False)))
        nucleotide_node.append(float(structure.get("modified", False)))

        # ------------------------------------------
        # Surface metrics
        # ------------------------------------------
        surface_metrics = node.get("surface_metrics", {})

        # Fractional Accessible Surface Area (FASA).
        fasa = surface_metrics.get("fasa", {})
        nucleotide_node.append(float(fasa.get("bs", 0.0)))
        nucleotide_node.append(float(fasa.get("pp", 0.0)))
        nucleotide_node.append(float(fasa.get("sg", 0.0)))
        nucleotide_node.append(float(fasa.get("sr", 0.0)))
        nucleotide_node.append(float(fasa.get("wg", 0.0)))

        # ------------------------------------------
        # Interaction metrics
        # ------------------------------------------
        interaction_metrics = node.get("interaction_metrics", {})

        nucleotide_node.append(float(interaction_metrics.get("interaction_count", 0.0)))

        # ------------------------------------------
        # Electrostatics properties
        # ------------------------------------------
        electrostatics = node.get("electrostatics", {})

        nucleotide_node.append(float(electrostatics.get("partial_charge", 0.0)))

        # Validate the correct length of the node.
        assert len(nucleotide_node) == DNA_NODE_DIM, \
            f"Dimension mismatch in DNA node feature length: {len(nucleotide_node)} != {DNA_NODE_DIM}"

        nodes_list.append(nucleotide_node)

    return nodes_list


# -----------------------------------------------------------------------------
# Protein Node Features
# -----------------------------------------------------------------------------
def protein_node_features(entry: Dict[str, Any], encoder: Dict[str, OneHotEncoder],
                          protein_data_map: Dict[str, Dict[str, Any]]) -> List[List[float]]:
    """
    Build the per-residue feature vectors.

    Each vector concatenates the one-hot amino acid identity, the one-hot
    secondary structure and the helicoidal coordinates, the surface curvature and
    SAP score, the FASA / SESA areas split into main chain and side chain, the
    interface interaction count and the aggregated partial charge.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.
        encoder (Dict[str, OneHotEncoder]): The categorical encoders.
        protein_data_map (Dict[str, Dict[str, Any]]): Amino acid nomenclature
            keyed by ID, used for the name one-hot.

    Returns:
        List[List[float]]: One feature vector per amino acid, in node order.
    """
    nodes_list = []

    nodes = entry.get("nodes", {})
    protein_nodes = nodes.get("protein", [])

    for node in protein_nodes:
        aminoacid_node = []

        id_key = str(node.get("id_key", "unknown"))
        aminoacid_info = protein_data_map.get(id_key, {})

        # ------------------------------------------
        # Identity
        # ------------------------------------------
        name = str(aminoacid_info.get("name", "UNK"))
        aminoacid_node += encoder["STANDARD_AMINOACID_MAP"].encode(name)

        # ------------------------------------------
        # Structure
        # ------------------------------------------
        structure = node.get("structure", {})

        secondary_structure = str(structure.get("secondary_structure", "unknown"))
        aminoacid_node += encoder["AMINOACID_SECONDARY_STRUCTURE_MAP"].encode(secondary_structure)

        helicoidal_coordinates = structure.get("helicoidal_coordinates", {})
        aminoacid_node.append(float(helicoidal_coordinates.get("phi", 0.0)))
        aminoacid_node.append(float(helicoidal_coordinates.get("s", 0.0)))
        aminoacid_node.append(float(helicoidal_coordinates.get("rho", 0.0)))

        # ------------------------------------------
        # Surface metrics
        # ------------------------------------------
        surface_metrics = node.get("surface_metrics", {})

        aminoacid_node.append(float(surface_metrics.get("cv_coarse", 0.0)))
        aminoacid_node.append(float(surface_metrics.get("cv_fine", 0.0)))
        aminoacid_node.append(float(surface_metrics.get("sap_score", 0.0)))

        # Fractional Accessible Surface Area (FASA).
        fasa = surface_metrics.get("fasa", {})
        aminoacid_node.append(float(fasa.get("mc", 0.0)))
        aminoacid_node.append(float(fasa.get("sc", 0.0)))

        # Solvent excluded surface area (SESA).
        sesa = surface_metrics.get("sesa", {})
        aminoacid_node.append(float(sesa.get("mc", 0.0)))
        aminoacid_node.append(float(sesa.get("sc", 0.0)))

        # ------------------------------------------
        # Interaction metrics
        # ------------------------------------------
        interaction_metrics = node.get("interaction_metrics", {})

        aminoacid_node.append(float(interaction_metrics.get("interaction_count", 0.0)))

        # ------------------------------------------
        # Electrostatics properties
        # ------------------------------------------
        electrostatics = node.get("electrostatics", {})

        aminoacid_node.append(float(electrostatics.get("partial_charge", 0.0)))

        # Validate the correct length of the node.
        assert len(aminoacid_node) == PROTEIN_NODE_DIM, \
            f"Dimension mismatch in protein node feature length: {len(aminoacid_node)} != {PROTEIN_NODE_DIM}"

        nodes_list.append(aminoacid_node)

    return nodes_list



# ==============================================================================
# EDGE EXTRACTION
# ==============================================================================
# Build the (edge_index, edge_attr) pair for every edge type. Endpoints are
# mapped through the node index maps.

# -----------------------------------------------------------------------------
# DNA Base-Stacking Edges
# -----------------------------------------------------------------------------
def dna_stack_edges(entry: Dict[str, Any], dna_index_map: Dict[str, int]) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Build the DNA base-stacking edges.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.
        dna_index_map (Dict[str, int]): Map from nucleotide ID key to row index.

    Returns:
        Tuple[List[List[int]], List[List[float]]]: The [source, target] index
            lists and the per-edge attribute vectors.
    """
    source_ids = []
    target_ids = []
    edge_attr = []

    edges = entry.get("edges", {})
    dna_stacks = edges.get("dna_stacks", [])

    for dna_stack in dna_stacks:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        source_id_key = str(dna_stack.get("source_id_key", "unknown"))
        target_id_key = str(dna_stack.get("target_id_key", "unknown"))

        if source_id_key not in dna_index_map or target_id_key not in dna_index_map:
            continue

        attributes = []

        # ------------------------------------------
        # Geometry
        # ------------------------------------------
        geometry = dna_stack.get("geometry", {})

        attributes.append(float(geometry.get("mindist", 3.5)))

        # ------------------------------------------
        # Shape parameters
        # ------------------------------------------
        shape_parameters = dna_stack.get("shape_parameters", {})
        attributes.append(float(shape_parameters.get("twist", 0.0)))
        attributes.append(float(shape_parameters.get("roll", 0.0)))
        attributes.append(float(shape_parameters.get("rise", 0.0)))
        attributes.append(float(shape_parameters.get("slide", 0.0)))
        attributes.append(float(shape_parameters.get("tilt", 0.0)))
        attributes.append(float(shape_parameters.get("shift", 0.0)))

        # Validate the correct length of the edge.
        assert len(attributes) == DNA_STACK_EDGES_DIM, \
            f"Dimension mismatch in DNA stack edge length: {len(attributes)} != {DNA_STACK_EDGES_DIM}"
        
        source_ids.append(dna_index_map[source_id_key])
        target_ids.append(dna_index_map[target_id_key])
        edge_attr.append(attributes)

    edge_index = [source_ids, target_ids]

    return edge_index, edge_attr


# -----------------------------------------------------------------------------
# DNA Backbone Edges
# -----------------------------------------------------------------------------
def dna_backbone_edges(entry: Dict[str, Any], dna_index_map: Dict[str, int]) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Build the DNA backbone (phosphodiester) edges.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.
        dna_index_map (Dict[str, int]): Map from nucleotide ID key to row index.

    Returns:
        Tuple[List[List[int]], List[List[float]]]: The [source, target] index
            lists and the per-edge attribute vectors.
    """
    source_ids = []
    target_ids = []
    edge_attr = []

    edges = entry.get("edges", {})
    dna_backbone = edges.get("dna_backbone", [])

    for dna_link in dna_backbone:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        source_id_key = str(dna_link.get("source_id_key", "unknown"))
        target_id_key = str(dna_link.get("target_id_key", "unknown"))

        if source_id_key not in dna_index_map or target_id_key not in dna_index_map:
            continue

        attributes = []

        # ------------------------------------------
        # Geometry
        # ------------------------------------------
        geometry = dna_link.get("geometry", {})
        attributes.append(float(geometry.get("link_distance", 1.6)))

        # Validate the correct length of the edge.
        assert len(attributes) == DNA_BACKBONE_EDGES_DIM, \
            f"Dimension mismatch in DNA backbone edge length: {len(attributes)} != {DNA_BACKBONE_EDGES_DIM}"

        source_ids.append(dna_index_map[source_id_key])
        target_ids.append(dna_index_map[target_id_key])
        edge_attr.append(attributes)

    edge_index = [source_ids, target_ids]

    return edge_index, edge_attr


# -----------------------------------------------------------------------------
# DNA Base-Pairing Edges
# -----------------------------------------------------------------------------
def dna_base_pair_edges(entry: Dict[str, Any], encoder: Dict[str, OneHotEncoder],
                        dna_index_map: Dict[str, int]) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Build the DNA base-pairing edges.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.
        encoder (Dict[str, OneHotEncoder]): The categorical encoders.
        dna_index_map (Dict[str, int]): Map from nucleotide ID key to row index.

    Returns:
        Tuple[List[List[int]], List[List[float]]]: The [source, target] index
            lists and the per-edge attribute vectors.
    """
    source_ids = []
    target_ids = []
    edge_attr = []

    edges = entry.get("edges", {})
    dna_base_pairs = edges.get("dna_base_pairs", [])

    for dna_base_pair in dna_base_pairs:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        source_id_key = str(dna_base_pair.get("source_id_key", "unknown"))
        target_id_key = str(dna_base_pair.get("target_id_key", "unknown"))

        if source_id_key not in dna_index_map or target_id_key not in dna_index_map:
            continue

        attributes = []

        # ------------------------------------------
        # Structure
        # ------------------------------------------
        structure = dna_base_pair.get("structure", {})
        pair_type = str(structure.get("pair_type", "unknown"))
        attributes += encoder["BASE_PAIR_TYPE_MAP"].encode(pair_type)

        attributes.append(float(structure.get("mismatched", False)))

        # ------------------------------------------
        # Chemistry
        # ------------------------------------------
        chemistry = dna_base_pair.get("chemistry", {})

        # Hydrogen bonds
        hbonds = chemistry.get("hbonds", {})
        attributes += reduce_distances(hbonds.get("distances", []))
        

        # ------------------------------------------
        # Shape parameters
        # ------------------------------------------
        shape_parameters = dna_base_pair.get("shape_parameters", {})
        attributes.append(float(shape_parameters.get("buckle", 0.0)))
        attributes.append(float(shape_parameters.get("propeller", 0.0)))
        attributes.append(float(shape_parameters.get("opening", 0.0)))
        attributes.append(float(shape_parameters.get("shear", 0.0)))
        attributes.append(float(shape_parameters.get("stagger", 0.0)))
        attributes.append(float(shape_parameters.get("stretch", 0.0)))

        # Validate the correct length of the edge.
        assert len(attributes) == DNA_BASE_PAIR_EDGES_DIM, \
            f"Dimension mismatch in DNA base pair edge length: {len(attributes)} != {DNA_BASE_PAIR_EDGES_DIM}"

        source_ids.append(dna_index_map[source_id_key])
        target_ids.append(dna_index_map[target_id_key])
        edge_attr.append(attributes)

    edge_index = [source_ids, target_ids]

    return edge_index, edge_attr


# -----------------------------------------------------------------------------
# Protein Backbone Edges
# -----------------------------------------------------------------------------
def protein_backbone_edges(entry: Dict[str, Any], protein_index_map: Dict[str, int]) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Build the protein backbone (peptide-bond) edges.

    The gap attribute is the only edge feature: it is reserved for chain breaks
    and, in the current standardized JSON, is always zero.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.
        protein_index_map (Dict[str, int]): Map from amino acid ID key to row
            index.

    Returns:
        Tuple[List[List[int]], List[List[float]]]: The [source, target] index
            lists and the per-edge attribute vectors.
    """
    source_ids = []
    target_ids = []
    edge_attr = []

    edges = entry.get("edges", {})
    protein_backbone = edges.get("protein_backbone", [])

    for protein_link in protein_backbone:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        source_id_key = str(protein_link.get("source_id_key", "unknown"))
        target_id_key = str(protein_link.get("target_id_key", "unknown"))

        if source_id_key not in protein_index_map or target_id_key not in protein_index_map:
            continue

        attributes = []

        # ------------------------------------------
        # Geometry
        # ------------------------------------------
        geometry = protein_link.get("geometry", {})
        attributes.append(float(geometry.get("gap", 0)))

        # Validate the correct length of the edge.
        assert len(attributes) == PROTEIN_BACKBONE_EDGES_DIM, \
            f"Dimension mismatch in protein backbone edge length: {len(attributes)} != {PROTEIN_BACKBONE_EDGES_DIM}"
        
        source_ids.append(protein_index_map[source_id_key])
        target_ids.append(protein_index_map[target_id_key])
        edge_attr.append(attributes)

    edge_index = [source_ids, target_ids]

    return edge_index, edge_attr


# -----------------------------------------------------------------------------
# Protein-DNA Interaction Edges
# -----------------------------------------------------------------------------
def protein_dna_interaction_edges(entry: Dict[str, Any], encoder: Dict[str, OneHotEncoder],
                                  protein_index_map: Dict[str, int],
                                  dna_index_map: Dict[str, int]) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Build the bipartite protein-DNA interaction edges (amino acid -> nucleotide).

    Args:
        entry (Dict[str, Any]): A single standardized complex model.
        encoder (Dict[str, OneHotEncoder]): The categorical encoders.
        protein_index_map (Dict[str, int]): Map from amino acid ID key to index.
        dna_index_map (Dict[str, int]): Map from nucleotide ID key to index.

    Returns:
        Tuple[List[List[int]], List[List[float]]]: The [source, target] index
            lists and the per-edge attribute vectors.
    """
    source_ids = []
    target_ids = []
    edge_attr = []

    edges = entry.get("edges", {})
    interactions = edges.get("protein_dna_interactions", [])

    for interaction in interactions:
        # ------------------------------------------
        # Identity
        # ------------------------------------------
        aminoacid_id_key = str(interaction.get("aminoacid_id_key", "unknown"))
        nucleotide_id_key = str(interaction.get("nucleotide_id_key", "unknown"))

        # Skip the edge if either endpoint is missing from the node tables.
        if aminoacid_id_key not in protein_index_map or nucleotide_id_key not in dna_index_map:
            continue

        attributes = []

        # ------------------------------------------
        # Geometry
        # ------------------------------------------
        geometry = interaction.get("geometry", {})

        attributes.append(float(geometry.get("min_distance", 0.0)))
        attributes.append(float(geometry.get("mean_nn_distance", 0.0)))
        attributes.append(float(geometry.get("cm_distance", 0.0)))

        geometry_type = str(geometry.get("geometry_type", "unknown"))
        attributes += encoder["INTERACTION_GEOMETRY_TYPE_MAP"].encode(geometry_type)

        # ------------------------------------------
        # Surface metrics
        # ------------------------------------------
        surface_metrics = interaction.get("surface_metrics", {})

        # Buried Accessible Surface Area (BASA).
        basa = surface_metrics.get("basa", {})
        for moiety in ["bs", "pp", "sg", "sr", "wg"]:
            moiety_data = basa.get(moiety, {})
            attributes.append(float(moiety_data.get("mc", 0.0)))
            attributes.append(float(moiety_data.get("sc", 0.0)))

        # ------------------------------------------
        # Chemistry
        # ------------------------------------------
        chemistry = interaction.get("chemistry", {})

        attributes.append(float(chemistry.get("is_weak_interaction", False)))

        # Hydrogen bonds.
        hbonds = chemistry.get("hbonds", {})
        hbonds_total = hbonds.get("total", {})
        for moiety in ["bs", "pp", "sg", "sr", "wg"]:
            moiety_data = hbonds_total.get(moiety, {})
            attributes.append(float(moiety_data.get("mc", 0)))
            attributes.append(float(moiety_data.get("sc", 0)))
        attributes += reduce_distances(hbonds.get("direct_distances", []))
        attributes += reduce_distances(hbonds.get("water_mediated_distances", []))

        # Van der Waals interactions.
        vdw_interactions = chemistry.get("vdw_interactions", {})
        vdw_interactions_total = vdw_interactions.get("total", {})
        for moiety in ["bs", "pp", "sg", "sr", "wg"]:
            moiety_data = vdw_interactions_total.get(moiety, {})
            attributes.append(float(moiety_data.get("mc", 0)))
            attributes.append(float(moiety_data.get("sc", 0)))
        attributes += reduce_distances(vdw_interactions.get("distances", []))

        # Validate the correct length of the edge.
        assert len(attributes) == PROTEIN_DNA_INTERACTION_EDGES_DIM, \
            f"Dimension mismatch in protein DNA interaction edge length: {len(attributes)} != {PROTEIN_DNA_INTERACTION_EDGES_DIM}"

        source_ids.append(protein_index_map[aminoacid_id_key])
        target_ids.append(dna_index_map[nucleotide_id_key])
        edge_attr.append(attributes)

    edge_index = [source_ids, target_ids]

    return edge_index, edge_attr



# ==============================================================================
# GLOBAL CONTEXT
# ==============================================================================
# Aggregate the per-protein-chain interface blocks and the DNA descriptors into a
# single fixed-length global feature vector.

# -----------------------------------------------------------------------------
# Global Context Features
# -----------------------------------------------------------------------------
def global_context_features(entry: Dict[str, Any], encoder: Dict[str, OneHotEncoder]) -> List[float]:
    """
    Aggregate the interface and DNA blocks into one fixed-length global vector.

    The standardized model stores one interface block per protein chain, one
    block per DNA entity and one per helical segment. Each field is collapsed
    with the reducer that matches its physical nature: extensive quantities are
    summed, intensive ones are averaged weighting by the interaction count (for
    the chains) or by the strand length (for the segments), flags are ORed and
    categories are resolved by weighted majority. The block counts and the
    experimental conditions are appended as raw scalars.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.
        encoder (Dict[str, OneHotEncoder]): The categorical encoders.

    Returns:
        List[float]: The aggregated global feature vector, of length
            GLOBAL_CONTEXT_DIM.
    """
    global_data = []

    global_context = entry.get("global_context", {})
    interfaces = global_context.get("interfaces", {})
    dna_descriptors = global_context.get("dna_descriptors", {})

    # ------------------------------------------
    # Conditions
    # ------------------------------------------
    global_data.append(float(global_context.get("resolution", 2.5)))
    global_data.append(float(global_context.get("temperature", 298.0)))
    global_data.append(float(global_context.get("pH", 7.4)))

    # ------------------------------------------
    # Interfaces
    # ------------------------------------------
    # Extract the number of interfaces first to define array shapes.
    num_interfaces = int(global_context.get("num_interfaces", 0))
    global_data.append(float(num_interfaces))

    # Interaction metrics
    metrics = interfaces.get("interaction_metrics", {})
    chain_weights = metrics.get("num_interactions", [0.0] * num_interfaces)
    
    global_data += reduce_intensive(metrics.get("hydrophobicity", [0.0] * num_interfaces), chain_weights)
    global_data += reduce_extensive(metrics.get("num_interactions", [0.0] * num_interfaces))
    global_data += reduce_extensive(metrics.get("num_weak_interactions", [0.0] * num_interfaces))

    # Protein surface geometry
    geometry = interfaces.get("protein_surface_geometry", {})

    cv_coarse = geometry.get("cv_coarse", {})
    for ratio in ("flat_ratio", "valley_ratio", "peak_ratio"):
        global_data += reduce_intensive(cv_coarse.get(ratio, [0.0] * num_interfaces), chain_weights)

    cv_fine = geometry.get("cv_fine", {})
    for ratio in ("flat_ratio", "valley_ratio", "peak_ratio"):
        global_data += reduce_intensive(cv_fine.get(ratio, [0.0] * num_interfaces), chain_weights)

    # Recognition propensities
    propensities = interfaces.get("aminoacid_propensities", {})

    for aminoacid in STANDARD_AMINOACID_MAP:
        global_data += reduce_intensive(propensities.get(aminoacid, [0.0] * num_interfaces), chain_weights)

    # ------------------------------------------
    # DNA entities
    # ------------------------------------------
    entities = dna_descriptors.get("entities", {})

    # Extract the number of entities first to define array shapes.
    num_entities = int(dna_descriptors.get("num_entities", 0))
    global_data.append(float(num_entities))

    # Topology.
    topology = entities.get("topology", {})
    entity_weights = topology.get("num_strands", [0.0] * num_entities)

    global_data += reduce_categorical(topology.get("entity_type", [0.0] * num_entities), encoder["DNA_ENTITY_TYPE_MAP"], entity_weights)
    global_data += reduce_extensive(topology.get("num_strands", [0.0] * num_entities))
    global_data += reduce_extensive(topology.get("num_helical_segments", [0.0] * num_entities))
    global_data += reduce_extensive(topology.get("num_single_stranded_segments", [0.0] * num_entities))

    # Modifications.
    modifications = entities.get("modifications", {})
    global_data += reduce_flag(modifications.get("methylated_cytosine", [0.0] * num_entities))
    global_data += reduce_flag(modifications.get("non_standard_nucleotides", [0.0] * num_entities))

    # ------------------------------------------
    # DNA helical segments
    # ------------------------------------------
    helical_segments = dna_descriptors.get("helical_segments", {})

    # Extract the number of helical segments first to define array shapes.
    num_helical_segments = int(dna_descriptors.get("num_helical_segments", 0))
    global_data.append(float(num_helical_segments))

    # Sequence.
    sequence = helical_segments.get("sequence", {})
    strand_lengths = sequence.get("length", [0.0] * num_helical_segments)

    global_data += reduce_extensive(sequence.get("length", [0.0] * num_helical_segments))
    global_data += reduce_intensive(sequence.get("GC_content", []), strand_lengths)

    # Classification.
    global_data += reduce_categorical(helical_segments.get("classification", [0.0] * num_helical_segments), encoder["DNA_HELICAL_SEGMENT_CLASSIFICATION_MAP"], strand_lengths)

    # Geometry.
    geometry = helical_segments.get("geometry", {})

    global_data += reduce_intensive(geometry.get("mean_radius", []), strand_lengths)
    global_data += reduce_categorical(geometry.get("axis_curvature", []), encoder["DNA_HELICAL_SEGMENT_AXIS_CURVATURE_MAP"], strand_lengths)
    global_data += reduce_extensive(geometry.get("axis_length", []))

    # Content.
    content = helical_segments.get("content", {})

    global_data += reduce_flag(content.get("a_tracts", []))
    global_data += reduce_flag(content.get("hoogsteen_pairs", []))
    global_data += reduce_flag(content.get("mismatches", []))
    global_data += reduce_flag(content.get("non_wc_pairs", []))
   

    # Validate the correct length of the global context.
    assert len(global_data) == GLOBAL_CONTEXT_DIM, \
            f"Dimension mismatch in global context length: {len(global_data)} != {GLOBAL_CONTEXT_DIM}"
    
    return global_data



# ==============================================================================
# GRAPH ASSEMBLY
# ==============================================================================
# Turn the extracted node, edge, global and target blocks into a single
# HeteroData graph.

# -----------------------------------------------------------------------------
# Instantiate Heterogeneous Graph
# -----------------------------------------------------------------------------
def instantiate_hetero_data(entry: Dict[str, Any], encoder: Dict[str, OneHotEncoder],
                            esm2_map: Dict[str, torch.Tensor]) -> HeteroData:
    """
    Assemble a single complex into a HeteroData graph.

    Builds the DNA and protein node features (plus the protein ESM-2 matrix), the
    four edge types (with the protein-DNA interaction edge mirrored in both
    directions), the global interface-context vector and the pKd target.

    Args:
        entry (Dict[str, Any]): A single standardized complex model.
        encoder (Dict[str, OneHotEncoder]): The categorical encoders.
        esm2_map (Dict[str, torch.Tensor]): Map from residue ID key to its ESM-2
            embedding, from load_esm2_embeddings.

    Returns:
        HeteroData: The assembled heterogeneous graph.
    """
    # ------------------------------------------
    # Extractions
    # ------------------------------------------
    # Basic information.
    dna_index_map, dna_info_map = dna_node_info(entry)
    protein_index_map, protein_info_map = protein_node_info(entry)

    # Node features.
    dna_nodes = dna_node_features(entry, encoder, dna_info_map)
    protein_nodes = protein_node_features(entry, encoder, protein_info_map)

    # Edge features.
    dna_base_pair_edge_index, dna_base_pair_edge_attr = dna_base_pair_edges(entry, encoder, dna_index_map)
    dna_stack_edge_index, dna_stack_edge_attr = dna_stack_edges(entry, dna_index_map)
    dna_backbone_edge_index, dna_backbone_edge_attr = dna_backbone_edges(entry, dna_index_map)
    protein_backbone_edge_index, protein_backbone_edge_attr = protein_backbone_edges(entry, protein_index_map)
    protein_dna_interaction_edge_index, protein_dna_interaction_edge_attr = protein_dna_interaction_edges(entry, encoder, protein_index_map, dna_index_map)

    # Global context.
    global_context_data = global_context_features(entry, encoder)

    # ------------------------------------------
    # Generate HeteroData
    # ------------------------------------------
    data = HeteroData()

    # ID.
    data.pdb_id_key = str(entry.get("pdb_id", "unknown")).lower()

    # Node features.
    data["protein"].x = torch.tensor(protein_nodes, dtype=torch.float32)
    data["dna"].x = torch.tensor(dna_nodes, dtype=torch.float32)

    # Structural classes.
    classes = entry.get("classes", {})
    data.dna_class = str(classes.get("dna", "unknown"))

    # ESM-2.
    # Align the ESM-2 matrix with the protein node rows, defaulting to zeros for
    # residues without an embedding.
    protein_node_list = entry.get("nodes", {}).get("protein", [])
    esm2_list = [None] * len(protein_node_list)

    for index, node in enumerate(protein_node_list):
        vector = esm2_map.get(str(node.get("id_key", "unknown")))
        if vector is None:
            esm2_list[index] = torch.zeros(ESM_DIM, dtype=torch.float32)
        else:
            esm2_list[index] = torch.as_tensor(vector, dtype=torch.float32).view(-1)[:ESM_DIM]
    
    # Aggregate the ESM2 values to the data.
    if esm2_list:
        data["protein"].esm2 = torch.stack(esm2_list, dim=0)
    else:
        data["protein"].esm2 = torch.zeros((0, ESM_DIM), dtype=torch.float32)

    assert data["protein"].esm2.size(0) == data["protein"].x.size(0), (
        f"ESM-2 row mismatch in {data.pdb_id_key}: "
        f"{data['protein'].esm2.size(0)} =! {data['protein'].x.size(0)}"
    )

    # ---------------------------------------------------------------------
    # Edge Index Helper
    # ---------------------------------------------------------------------
    def index_tensor(edge_index: List[List[int]]) -> torch.Tensor:
        """
        Cast an [source, target] index pair into a (2, num_edges) long tensor.

        An edge type with no edges is shaped explicitly, so the collate function
        can still stack it alongside the dense graphs of the batch.

        Args:
            edge_index (List[List[int]]): The [source, target] index lists.

        Returns:
            torch.Tensor: The edge index tensor.
        """
        if len(edge_index[0]) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(edge_index, dtype=torch.long)

    # ---------------------------------------------------------------------
    # Edge Attribute Helper
    # ---------------------------------------------------------------------
    def attr_tensor(edge_attr: List[List[float]], dim: int) -> torch.Tensor:
        """
        Cast a list of edge-attribute vectors into a (num_edges, dim) tensor.

        Args:
            edge_attr (List[List[float]]): The per-edge attribute vectors.
            dim (int): The expected attribute length, used to shape the empty
                case so the feature width stays consistent across the batch.

        Returns:
            torch.Tensor: The edge attribute tensor.
        """
        if len(edge_attr) == 0:
            return torch.zeros((0, dim), dtype=torch.float32)
        return torch.tensor(edge_attr, dtype=torch.float32)

    # Edge features.
    data["dna", "base_pair", "dna"].edge_index = index_tensor(dna_base_pair_edge_index)
    data["dna", "base_pair", "dna"].edge_attr = attr_tensor(dna_base_pair_edge_attr, DNA_BASE_PAIR_EDGES_DIM)
    data["dna", "rev_base_pair", "dna"].edge_index = index_tensor(dna_base_pair_edge_index).flip(0)
    data["dna", "rev_base_pair", "dna"].edge_attr = attr_tensor(dna_base_pair_edge_attr, DNA_BASE_PAIR_EDGES_DIM).clone()

    data["dna", "stack", "dna"].edge_index = index_tensor(dna_stack_edge_index)
    data["dna", "stack", "dna"].edge_attr = attr_tensor(dna_stack_edge_attr, DNA_STACK_EDGES_DIM)
    data["dna", "rev_stack", "dna"].edge_index = index_tensor(dna_stack_edge_index).flip(0)
    data["dna", "rev_stack", "dna"].edge_attr = attr_tensor(dna_stack_edge_attr, DNA_STACK_EDGES_DIM).clone()

    data["dna", "backbone", "dna"].edge_index = index_tensor(dna_backbone_edge_index)
    data["dna", "backbone", "dna"].edge_attr = attr_tensor(dna_backbone_edge_attr, DNA_BACKBONE_EDGES_DIM)
    data["dna", "rev_backbone", "dna"].edge_index = index_tensor(dna_backbone_edge_index).flip(0)
    data["dna", "rev_backbone", "dna"].edge_attr = attr_tensor(dna_backbone_edge_attr, DNA_BACKBONE_EDGES_DIM).clone()

    # Although the information only travels 5' -> 3', the reverse direction is
    # kept as its own relation so the model can learn to weight it down.
    data["protein", "backbone", "protein"].edge_index = index_tensor(protein_backbone_edge_index)
    data["protein", "backbone", "protein"].edge_attr = attr_tensor(protein_backbone_edge_attr, PROTEIN_BACKBONE_EDGES_DIM)
    data["protein", "rev_backbone", "protein"].edge_index = index_tensor(protein_backbone_edge_index).flip(0)
    data["protein", "rev_backbone", "protein"].edge_attr = attr_tensor(protein_backbone_edge_attr, PROTEIN_BACKBONE_EDGES_DIM).clone()
    
    data["protein", "interaction", "dna"].edge_index = index_tensor(protein_dna_interaction_edge_index)
    data["protein", "interaction", "dna"].edge_attr = attr_tensor(protein_dna_interaction_edge_attr, PROTEIN_DNA_INTERACTION_EDGES_DIM)
    data["dna", "rev_interaction", "protein"].edge_index = index_tensor(protein_dna_interaction_edge_index).flip(0)
    data["dna", "rev_interaction", "protein"].edge_attr = attr_tensor(protein_dna_interaction_edge_attr, PROTEIN_DNA_INTERACTION_EDGES_DIM).clone()
    
    # Global context.
    data.global_context = torch.tensor([global_context_data], dtype=torch.float32)

    return data


# -----------------------------------------------------------------------------
# Build Graph
# -----------------------------------------------------------------------------
def build_graph(json_path: str, esm2_path: str, encoder: Dict[str, OneHotEncoder]) -> HeteroData:
    """
    Assemble the heterogeneous graph of one complex from its two input files.

    Args:
        json_path (str): Path to the standardized JSON file of the complex.
        esm2_path (str): Path to the ESM-2 .pt file of the complex.
        encoder (Dict[str, OneHotEncoder]): The categorical encoders.

    Returns:
        HeteroData: The assembled graph.
    """
    entry = read_entry(json_path)

    if not entry:
        raise SystemExit("[ERROR] No valid data found in the standardized JSON. Aborting.")

    esm2_map = load_esm2_embedding(esm2_path)

    if not esm2_map:
        raise SystemExit("[ERROR] No valid ESM-2 embeddings found. Aborting.")

    graph = instantiate_hetero_data(entry, encoder, esm2_map)

    return graph



# ==============================================================================
# FILE I/O
# ==============================================================================
# This section ingests the two mandatory inputs of the pipeline: the
# standardized JSON of one complex (nodes, edges and global context, as
# produced upstream of this script) and its ESM-2 per-residue embeddings.

# -----------------------------------------------------------------------------
# Read Standardized Entry
# -----------------------------------------------------------------------------
def read_entry(json_path: str) -> Dict[str, Any]:
    """
    Load a single-complex standardized JSON file.

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
# Load ESM-2 Embedding
# -----------------------------------------------------------------------------
def load_esm2_embedding(esm2_path: str) -> Dict[str, torch.Tensor]:
    """
    Load the per-residue ESM-2 embeddings.

    Args:
        esm2_path (str): Path to the .pt file produced by compute_esm2.py.

    Returns:
        Dict[str, torch.Tensor]: Map from residue id_key to its embedding
            vector, or an empty dictionary if the file is missing or
            unreadable.
    """
    if not os.path.exists(esm2_path):
        print(f"[ERROR] ESM-2 embeddings not found at: {esm2_path}")
        return {}

    if os.path.isdir(esm2_path):
        print(f"[ERROR] Expected a .pt file, but a directory was found at: {esm2_path}")
        return {}

    try:
        raw_embeddings = torch.load(esm2_path, map_location="cpu", weights_only=True)

        if not isinstance(raw_embeddings, dict):
            print(f"[ERROR] Unexpected ESM-2 payload at {esm2_path}: "
                  f"expected a dict, got {type(raw_embeddings).__name__}.")
            return {}

        # Cast back to float32; compute_esm2.py may have stored float16 (--half).
        return {str(id_key).strip(): vector.float() for id_key, vector in raw_embeddings.items()}

    except Exception as e:
        print(f"[ERROR] Could not load ESM-2 embeddings from {esm2_path}: {e}")
        return {}



# ==============================================================================
# NORMALISER ROUTING
# ==============================================================================
# Which normaliser of the saved bundle applies to which relation. The reverse
# relations hold a clone of the attributes of their forward counterpart, so they
# are scaled with the very same normaliser, exactly as during training.

# -----------------------------------------------------------------------------
# Apply Normalisers
# -----------------------------------------------------------------------------
def apply_normalizers(graph: HeteroData, normalizers: Dict[str, Any]) -> HeteroData:
    """
    Scale one graph with a fitted normaliser bundle.

    This is the transform half of normalise_split, with the fitting removed: the
    statistics of a new complex must never touch the scaling, or the prediction
    would depend on which other complexes happen to be in the same run.

    The target normaliser is deliberately not applied here. It is only used
    afterwards, to bring the prediction of this fold back to pKd.

    Args:
        graph (HeteroData): The raw assembled graph.
        normalizers (Dict[str, Any]): The bundle saved by save_normalizers.

    Returns:
        HeteroData: A scaled copy of the graph.
    """
    graph = copy.deepcopy(graph)

    # ------------------------------------------
    # Nodes
    # ------------------------------------------
    if graph["protein"].x.numel() > 0:
        graph["protein"].x = normalizers["protein"].transform(graph["protein"].x)

    if graph["dna"].x.numel() > 0:
        graph["dna"].x = normalizers["dna"].transform(graph["dna"].x)

    # ------------------------------------------
    # Edges
    # ------------------------------------------
    if graph["dna", "base_pair", "dna"].edge_attr.numel() > 0:
        graph["dna", "base_pair", "dna"].edge_attr = normalizers["dna_base_pair"].transform(graph["dna", "base_pair", "dna"].edge_attr)
        graph["dna", "rev_base_pair", "dna"].edge_attr = normalizers["dna_base_pair"].transform(graph["dna", "rev_base_pair", "dna"].edge_attr)

    if graph["dna", "stack", "dna"].edge_attr.numel() > 0:
        graph["dna", "stack", "dna"].edge_attr = normalizers["dna_stack"].transform(graph["dna", "stack", "dna"].edge_attr)
        graph["dna", "rev_stack", "dna"].edge_attr = normalizers["dna_stack"].transform(graph["dna", "rev_stack", "dna"].edge_attr)

    if graph["dna", "backbone", "dna"].edge_attr.numel() > 0:
        graph["dna", "backbone", "dna"].edge_attr = normalizers["dna_backbone"].transform(graph["dna", "backbone", "dna"].edge_attr)
        graph["dna", "rev_backbone", "dna"].edge_attr = normalizers["dna_backbone"].transform(graph["dna", "rev_backbone", "dna"].edge_attr)

    if graph["protein", "backbone", "protein"].edge_attr.numel() > 0:
        graph["protein", "backbone", "protein"].edge_attr = normalizers["protein_backbone"].transform(graph["protein", "backbone", "protein"].edge_attr)
        graph["protein", "rev_backbone", "protein"].edge_attr = normalizers["protein_backbone"].transform(graph["protein", "rev_backbone", "protein"].edge_attr)

    if graph["protein", "interaction", "dna"].edge_attr.numel() > 0:
        graph["protein", "interaction", "dna"].edge_attr = normalizers["protein_dna_interaction"].transform(graph["protein", "interaction", "dna"].edge_attr)
        graph["dna", "rev_interaction", "protein"].edge_attr = normalizers["protein_dna_interaction"].transform(graph["dna", "rev_interaction", "protein"].edge_attr)
                

    # ------------------------------------------
    # Global context
    # ------------------------------------------
    if graph.global_context.numel() > 0:
        graph.global_context = normalizers["global_context"].transform(graph.global_context)

    return graph


# -----------------------------------------------------------------------------
# Normalizer
# -----------------------------------------------------------------------------
class FeatureNormaliser:
    """
    Per-feature normaliser for continuous features, fitted on train only.
    """
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, indexes: List[int], skew_threshold: float = 1.0) -> None:
        """
        Initialize the normaliser.

        Args:
            indexes (List[int]): The column indices of the continuous features to be normalized.
            skew_threshold (float, optional): The absolute skewness threshold above which 
                robust scaling (median/IQR) is used instead of standard scaling (mean/std). 
                Defaults to 1.0.
        """
        self.indexes = list(indexes)
        self.skew_threshold = skew_threshold

        self.method = {}
        self.center = {}
        self.scale = {}
        self.fitted = False

    # -------------------------------------------------------------------------
    # Transform Features
    # -------------------------------------------------------------------------
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the fitted normalisation parameters to a given tensor.

        Args:
            x (torch.Tensor): The input feature tensor to be normalized.

        Returns:
            torch.Tensor: A new tensor with the continuous features normalized.
        """
        if not self.fitted or x.numel() == 0:
            return x

        transformed_tensor = x.clone()

        for column in self.indexes:
            center = self.center.get(column, 0.0)
            scale = self.scale.get(column, 1.0)

            # Safeguard against zero-division during transformation.
            if scale == 0.0:
                scale = 1.0

            transformed_tensor[:, column] = (transformed_tensor[:, column] - center) / scale
            
        return transformed_tensor
    
    # -------------------------------------------------------------------------
    # Inverse Transform Features
    # -------------------------------------------------------------------------
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalisation, mapping normalised values back to their original
        scale. Used to bring model predictions (and normalised targets) back to
        physical units (e.g. pKd) before computing metrics or plotting.

        Args:
            x (torch.Tensor): A tensor previously produced by transform (or a
                model prediction living in the same normalised space).

        Returns:
            torch.Tensor: A new tensor with the continuous features de-normalised.
        """
        if not self.fitted or x.numel() == 0:
            return x

        restored_tensor = x.clone()

        for column in self.indexes:
            center = self.center.get(column, 0.0)
            scale = self.scale.get(column, 1.0)

            # Mirror the safeguard used in transform.
            if scale == 0.0:
                scale = 1.0

            restored_tensor[:, column] = restored_tensor[:, column] * scale + center

        return restored_tensor

    # -------------------------------------------------------------------------
    # Restore Normaliser values
    # -------------------------------------------------------------------------
    def load_state_dict(self, state: Dict[str, Any]) -> "FeatureNormaliser":
        """
        Restore a normaliser previously exported with state_dict.

        Args:
            state (Dict[str, Any]): The dictionary produced by state_dict.

        Returns:
            FeatureNormaliser: The restored normaliser instance (self).
        """
        self.indexes = list(state.get("indexes", []))
        self.skew_threshold = state.get("skew_threshold", 1.0)
        self.method = {int(key): value for key, value in state.get("method", {}).items()}
        self.center = {int(key): float(value) for key, value in state.get("center", {}).items()}
        self.scale  = {int(key): float(value) for key, value in state.get("scale", {}).items()}
        self.fitted = bool(state.get("fitted", False))
        
        return self


# -----------------------------------------------------------------------------
# Normaliser Bundle Load
# -----------------------------------------------------------------------------
def load_normalizers(path: str) -> Dict[str, "FeatureNormaliser"]:
    """
    Restore a bundle of normalisers previously saved with save_normalizers.

    Args:
        path (str): Path to the saved normaliser bundle.

    Returns:
        Dict[str, FeatureNormaliser]: Map from name to restored normaliser.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)

    return {
        name: FeatureNormaliser(state.get("indexes", [])).load_state_dict(state)
        for name, state in payload.items()
    }



# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
# Encoder modules and the final predictor. Every block is kept independent so
# to_hetero can replicate the message-passing encoder per node and edge type,
# and so the three side channels (ESM-2, edge attributes and global context) can
# be projected before they reach the graph.

# -----------------------------------------------------------------------------
# Node Encoder Class
# -----------------------------------------------------------------------------
class NodeEncoder(nn.Module):
    """
    Two-block attentional message-passing encoder shared by both node types.

    A lazy input projection maps the DNA and protein feature vectors (whose
    widths differ, and whose protein width also carries the injected ESM-2
    channels) onto a common hidden size, which is what lets to_hetero replicate
    the module per node type. Each block is pre-normalized, runs a GATv2
    convolution conditioned on the encoded edge attributes, refines the message
    with a linear layer and merges it into the previous state through a GRU cell.
    That gated update, instead of a plain residual sum, is what keeps two rounds
    of neighbourhood mixing from over-smoothing the node states.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, hidden_channels: int, edge_dim: int, heads: int = 4,
                 proj_dropout_rate: float = 0.5, conv_dropout_rate: float = 0.5) -> None:
        """
        Build the input projection and the two message-passing blocks.

        Args:
            hidden_channels (int): Width of the shared hidden representation.
            edge_dim (int): Width of the encoded edge attributes.
            heads (int, optional): Number of attention heads. Defaults to 4.
            proj_dropout_rate (float, optional): Dropout applied after the input
                projection. Defaults to 0.5.
            conv_dropout_rate (float, optional): Dropout applied to every
                message. Defaults to 0.5.
        """
        super().__init__()

        # ------------------------------------------
        # Input projection
        # ------------------------------------------
        self.input_proj = nn.LazyLinear(out_features=hidden_channels)
        self.input_norm = nn.LayerNorm(normalized_shape=hidden_channels)
        self.proj_dropout = nn.Dropout(p=proj_dropout_rate)

        # ------------------------------------------
        # Layer 1
        # ------------------------------------------
        # add_self_loops=False is mandatory here: the bipartite relations created
        # by to_hetero have different source and target node types.
        self.conv1 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=edge_dim, heads=heads, concat=False, add_self_loops=False)
        self.norm1 = nn.LayerNorm(normalized_shape=hidden_channels)
        self.message_mlp1 = nn.Linear(hidden_channels, hidden_channels)
        self.conv_dropout1 = nn.Dropout(p=conv_dropout_rate)
        self.gru1 = nn.GRUCell(input_size=hidden_channels, hidden_size=hidden_channels)

        # ------------------------------------------
        # Layer 2
        # ------------------------------------------
        self.conv2 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=edge_dim, heads=heads, concat=False, add_self_loops=False)
        self.norm2 = nn.LayerNorm(normalized_shape=hidden_channels)
        self.message_mlp2 = nn.Linear(hidden_channels, hidden_channels)
        self.conv_dropout2 = nn.Dropout(p=conv_dropout_rate)
        self.gru2 = nn.GRUCell(input_size=hidden_channels, hidden_size=hidden_channels)

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Run the two rounds of message passing.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge index of the relation.
            edge_attr (torch.Tensor): The encoded edge attributes.

        Returns:
            torch.Tensor: The updated node representations.
        """
        # Input projection (post-norm: it is not a residual block).
        h0 = self.input_proj(x)
        h0 = self.input_norm(h0)
        h0 = F.gelu(h0)
        h0 = self.proj_dropout(h0)

        # Layer 1 (pre-norm inside the residual, GRU as the gated update).
        m1 = self.conv1(self.norm1(h0), edge_index, edge_attr)
        m1 = F.gelu(m1)
        m1 = self.conv_dropout1(m1)

        m1 = self.message_mlp1(m1)
        m1 = F.gelu(m1)

        h1 = self.gru1(m1, h0)

        # Layer 2.
        m2 = self.conv2(self.norm2(h1), edge_index, edge_attr)
        m2 = F.gelu(m2)
        m2 = self.conv_dropout2(m2)

        m2 = self.message_mlp2(m2)
        m2 = F.gelu(m2)
        
        h2 = self.gru2(m2, h1)

        return h2


# -----------------------------------------------------------------------------
# ESM-2 Encoder Class
# -----------------------------------------------------------------------------
class ESMEncoder(nn.Module):
    """
    Project and refine the per-residue ESM-2 embeddings.

    The 1280-dimensional language-model vector is compressed to a much narrower
    width and refined by a transformer-style feed-forward block. Two views come
    out of it: the projection itself, which feeds the attention read-out, and a
    normalized copy scaled by a learnable factor initialized at zero, which is
    the one concatenated to the protein nodes. Starting that injection at zero
    means message passing begins from the purely structural signal and only leans
    on the evolutionary one as far as training finds it useful.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, esm_projected_dim: int, dropout_rate: float, hidden_multiplier: int) -> None:
        """
        Build the projection, the refinement block and the injection gate.

        Args:
            esm_dim (int): Width of the raw ESM-2 embedding.
            esm_projected_dim (int): Width of the projected representation.
            dropout_rate (float): Dropout applied inside the refinement block.
            hidden_multiplier (int, optional): Expansion factor of the hidden
                layer. Defaults to 4 (standard Transformer FFN).
        """
        super().__init__()
        hidden_channels = esm_projected_dim * hidden_multiplier

        # ------------------------------------------
        # Dimensionality reduction
        # ------------------------------------------
        self.input_norm = nn.LayerNorm(ESM_DIM)
        self.input_proj = nn.Linear(ESM_DIM, esm_projected_dim)

        # ------------------------------------------
        # Layer 1
        # ------------------------------------------
        self.norm1 = nn.LayerNorm(esm_projected_dim)
        self.lin1 = nn.Linear(esm_projected_dim, hidden_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

        # ------------------------------------------
        # Layer 2
        # ------------------------------------------
        self.lin2 = nn.Linear(hidden_channels, esm_projected_dim)

        # ------------------------------------------
        # Injection gate
        # ------------------------------------------
        self.injection_norm = nn.LayerNorm(esm_projected_dim)
        self.injection_scale = nn.Parameter(torch.zeros(esm_projected_dim))

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, protein_esm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project the embeddings and build their injectable view.

        Args:
            protein_esm (torch.Tensor): The raw per-residue ESM-2 matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The projected embeddings and the
                scaled copy to be concatenated to the protein node features.
        """
        # Controlled dimensionality reduction.
        h0 = self.input_norm(protein_esm)
        h0 = self.input_proj(h0)
        
        # Non-linear refinement 1.
        m1 = self.norm1(h0)
        m1 = self.lin1(m1)

        m1 = F.gelu(m1)
        m1 = self.dropout(m1)

        # Non-linear refinement 2.
        m2 = self.lin2(m1)
        
        # Residual connection: preserves the direct signal of the projection.
        esm_proj = h0 + m2
        esm_injection = self.injection_scale * self.injection_norm(esm_proj)
        
        return esm_proj, esm_injection


# -----------------------------------------------------------------------------
# Edge Encoder Class
# -----------------------------------------------------------------------------
class EdgeEncoder(nn.Module):
    """
    Two-layer MLP that maps an edge-attribute vector onto the shared edge width.

    Every relation gets its own instance because their raw widths differ (from a
    single backbone distance to the 46 columns of a protein-DNA contact); a lazy
    first layer absorbs that per-type input size and the common output width is
    what GATv2Conv expects as edge_dim.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, edge_dim: int, dropout_rate: float, hidden_multiplier: int) -> None:
        """
        Build the two projection layers.

        Args:
            edge_dim (int): Width of the encoded edge attributes.
            dropout_rate (float): Dropout applied between both layers.
            hidden_multiplier (int, optional): Expansion factor of the hidden
                layer. Defaults to 2.
        """
        super().__init__()
        hidden_channels = edge_dim * hidden_multiplier

        # ------------------------------------------
        # Layer 1
        # ------------------------------------------
        self.lin1 = nn.LazyLinear(hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

        # ------------------------------------------
        # Layer 2
        # ------------------------------------------
        self.lin2 = nn.Linear(hidden_channels, edge_dim)
        self.norm2 = nn.LayerNorm(edge_dim)

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Encode one edge-attribute matrix.

        Args:
            edge_attr (torch.Tensor): The raw edge attributes of one relation.

        Returns:
            torch.Tensor: The encoded edge attributes.
        """
        # Layer 1.
        h = self.lin1(edge_attr)
        h = self.norm1(h)
            
        h = F.gelu(h)
        h = self.dropout(h)
        
        # Layer 2.
        h = self.lin2(h)
        h = self.norm2(h)
            
        h = F.gelu(h)
        
        return h


# -----------------------------------------------------------------------------
# Global Context Encoder Class
# -----------------------------------------------------------------------------
class GlobalContextEncoder(nn.Module):
    """
    Two-layer MLP that compresses the global context vector.

    The aggregated interface, DNA and conditions descriptors are far wider than
    the pooled graph embedding, so they are squeezed first: otherwise the head
    would see a vector dominated by the global block and would learn to ignore
    the message passing.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, projected_dim: int, dropout_rate: float, hidden_multiplier: int) -> None:
        """
        Build the two projection layers.

        Args:
            projected_dim (int): Width of the compressed representation.
            dropout_rate (float): Dropout applied between both layers.
            hidden_multiplier (int, optional): Expansion factor of the hidden
                layer. Defaults to 4.
        """
        super().__init__()
        hidden_channels = projected_dim * hidden_multiplier

        # ------------------------------------------
        # Layer 1
        # ------------------------------------------
        self.lin1 = nn.LazyLinear(out_features=hidden_channels)
        self.norm1 = nn.LayerNorm(normalized_shape=hidden_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

        # ------------------------------------------
        # Layer 2
        # ------------------------------------------
        self.lin2 = nn.Linear(in_features=hidden_channels, out_features=projected_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=projected_dim)

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, global_context: torch.Tensor) -> torch.Tensor:
        """
        Encode the per-graph global context vector.

        Args:
            global_context (torch.Tensor): The raw global context matrix, with
                one row per graph in the batch.

        Returns:
            torch.Tensor: The compressed global representation.
        """
        # Layer 1.
        h1 = self.lin1(global_context)
        h1 = self.norm1(h1)

        h1 = F.gelu(h1)
        h1 = self.dropout(h1)
        
        # Layer 2.
        h2 = self.lin2(h1)
        h2 = self.norm2(h2)
        
        return h2


# -----------------------------------------------------------------------------
# Linear Transformation Class
# -----------------------------------------------------------------------------
class LinearTransformation(nn.Module):
    """
    Regression head mapping the pooled graph embedding to the predicted pKd.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, in_features: int, hidden_channels: int, out_channels: int, dropout_rate: float) -> None:
        """
        Build the hidden layer and the output layer.

        Args:
            in_features (int): Width of the concatenated graph embedding.
            hidden_channels (int): Width of the hidden layer.
            out_channels (int): Number of predicted values.
            dropout_rate (float): Dropout applied before the output layer.
        """
        super().__init__()

        # ------------------------------------------
        # Layer 1
        # ------------------------------------------
        self.lin1 = nn.Linear(in_features, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        # ------------------------------------------
        # Layer 2
        # ------------------------------------------
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the target from the graph embedding.

        Args:
            x (torch.Tensor): The concatenated graph embedding.

        Returns:
            torch.Tensor: The predicted values.
        """
        # Layer 1.
        x = self.lin1(x)
        x = self.norm1(x)

        x = F.gelu(x)
        x = self.dropout1(x)
        
        # Layer 2.
        x = self.lin2(x)
        
        return x


# -----------------------------------------------------------------------------
# Affinity Predictor Class
# -----------------------------------------------------------------------------
class AffinityPredictor(torch.nn.Module):
    """
    End-to-end model predicting the binding affinity (pKd) of a complex.

    The graph embedding is assembled from four read-outs: an attentional pooling
    of the protein nodes, the same for the DNA nodes, a context-aware pooling of
    the ESM-2 projections, and the compressed global context. Concatenating them
    keeps the three sources of signal (local structure, evolutionary conservation
    and complex-level descriptors) separable at the head instead of forcing the
    message passing to carry all of them.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, 
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 out_channels: int,
                 # ------------------------------------------
                 # Node (GNN) parameters
                 # ------------------------------------------
                 node_hidden_dim: int,
                 node_heads: int,
                 node_proj_dropout: float,
                 node_conv_dropout: float,
                 # ------------------------------------------
                 # Edge parameters
                 # ------------------------------------------
                 edge_dim: int,
                 edge_hidden_multiplier: int,
                 edge_dropout: float,
                 # ------------------------------------------
                 # ESM parameters
                 # ------------------------------------------
                 esm_projected_dim: int,
                 esm_hidden_multiplier: int,
                 esm_dropout: float,
                 # ------------------------------------------
                 # Global context parameters
                 # ------------------------------------------
                 global_projected_dim: int,
                 global_hidden_multiplier: int,
                 global_dropout: float,
                 # ------------------------------------------
                 # Head (Readout) parameters
                 # ------------------------------------------
                 head_hidden_dim: int,
                 head_dropout: float):
        """
        Build every encoder, the heterogeneous GNN and the read-out layers.

        Args:
            metadata (Tuple): The HeteroData metadata (node and edge types).
            out_channels (int): Number of predicted values.
            
            node_hidden_dim (int): Width of the shared node representation.
            node_heads (int): Number of attention heads in the GNN.
            node_proj_dropout (float): Dropout for node input projection.
            node_conv_dropout (float): Dropout for GNN message passing.
            
            edge_dim (int): Width of the encoded edge attributes.
            edge_hidden_multiplier (int): Expansion factor for edge MLP.
            edge_dropout (float): Dropout applied to edge encoders.
            
            esm_dim (int): Width of the raw ESM-2 embedding.
            esm_projected_dim (int): Width of the projected ESM-2 embedding.
            esm_hidden_multiplier (int): Expansion factor for ESM refinement.
            esm_dropout (float): Dropout applied to ESM encoder.
            
            global_projected_dim (int): Width of the compressed global context.
            global_hidden_multiplier (int): Expansion factor for global MLP.
            global_dropout (float): Dropout applied to global context encoder.
            
            head_hidden_dim (int): Width of the hidden layer in the final MLP.
            head_dropout (float): Dropout applied before the final prediction.
        """
        super().__init__()
        
        # ------------------------------------------
        # ESM2 encoders
        # ------------------------------------------
        self.esm_encoder = ESMEncoder(esm_projected_dim=esm_projected_dim, dropout_rate=esm_dropout, hidden_multiplier=esm_hidden_multiplier
        )

        # ------------------------------------------
        # Edge encoders
        # ------------------------------------------
        self.edge_encoders = nn.ModuleDict()
        
        for edge_type in metadata[1]:
            key = "__".join(edge_type)
            self.edge_encoders[key] = EdgeEncoder(edge_dim=edge_dim, dropout_rate=edge_dropout, hidden_multiplier=edge_hidden_multiplier)

        # ------------------------------------------
        # Node encoder
        # ------------------------------------------
        node_encoder = NodeEncoder(
            hidden_channels=node_hidden_dim,
            edge_dim=edge_dim,
            heads=node_heads,
            proj_dropout_rate=node_proj_dropout,
            conv_dropout_rate=node_conv_dropout
        )
        self.gnn = to_hetero(node_encoder, metadata, aggr="sum")

        # ------------------------------------------
        # Global context encoder
        # ------------------------------------------
        self.global_encoder = GlobalContextEncoder(
            projected_dim=global_projected_dim, 
            dropout_rate=global_dropout,
            hidden_multiplier=global_hidden_multiplier
        )
        
        # ------------------------------------------
        # Readout layers
        # ------------------------------------------
        self.gate_protein = nn.Linear(node_hidden_dim, 1)
        self.pool_protein = AttentionalAggregation(gate_nn=self.gate_protein)
        
        self.gate_dna = nn.Linear(node_hidden_dim, 1)
        self.pool_dna = AttentionalAggregation(gate_nn=self.gate_dna)
        
        self.gate_esm = nn.Linear(node_hidden_dim + esm_projected_dim, 1)
        
        # ------------------------------------------
        # Final linear transformation
        # ------------------------------------------
        in_features = (node_hidden_dim * 2) + esm_projected_dim + global_projected_dim
        self.linear_transformation = LinearTransformation(
            in_features=in_features, 
            hidden_channels=head_hidden_dim, 
            out_channels=out_channels, 
            dropout_rate=head_dropout
        )
    
    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict, protein_esm, global_context):
        """
        Predict the affinity of every graph in the batch.

        Args:
            x_dict (Dict[str, torch.Tensor]): Node features per node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): Edge
                indexes per relation.
            edge_attr_dict (Dict[Tuple[str, str, str], torch.Tensor]): Raw edge
                attributes per relation.
            batch_dict (Dict[str, torch.Tensor]): Graph assignment of every node.
            protein_esm (torch.Tensor): The raw per-residue ESM-2 matrix.
            global_context (torch.Tensor): The raw global context, one row per
                graph.

        Returns:
            torch.Tensor: The predicted values, one row per graph.
        """
        # Shallow copy: never mutate the caller's dictionary.
        x_dict = dict(x_dict)

        # ------------------------------------------
        # Global context projection
        # ------------------------------------------
        global_proj = self.global_encoder(global_context)

        # ------------------------------------------
        # ESM-2 feature projection
        # ------------------------------------------
        esm_proj, esm_injection = self.esm_encoder(protein_esm)
        
        # Append the ESM-2 channels to the protein nodes.
        x_dict["protein"] = torch.cat([x_dict["protein"], esm_injection], dim=1)

        # ------------------------------------------
        # Edge feature projection
        # ------------------------------------------
        encoded_edge_attr = {}
        
        for edge_type, edge_attr in edge_attr_dict.items():
            key = "__".join(edge_type)
            encoded_edge_attr[edge_type] = self.edge_encoders[key](edge_attr)
            
        # ------------------------------------------
        # Message passing
        # ------------------------------------------
        x_dict = self.gnn(x_dict, edge_index_dict, encoded_edge_attr)

        # ------------------------------------------
        # Global Pooling
        # ------------------------------------------
        protein_pool = self.pool_protein(x_dict["protein"], batch_dict["protein"])
        dna_pool = self.pool_dna(x_dict["dna"], batch_dict["dna"])

        # Graphs with no protein nodes are discarded by the loader, so the last
        # row of the pooling always belongs to the last graph of the batch.
        num_graphs = protein_pool.size(0)

        # ------------------------------------------
        # ESM pooling
        # ------------------------------------------
        # What the node learnt from the graph is joined with its original
        # evolutionary nature, so the attention is computed on that enriched
        # context and the pooled ESM signal stays free of message-passing noise.
        esm_attention_features = torch.cat([x_dict["protein"], esm_proj], dim=1)
        
        alpha_logits_esm = self.gate_esm(esm_attention_features)
        alpha_esm = pyg_softmax(alpha_logits_esm, batch_dict["protein"])
        
        esm_pool = global_add_pool(alpha_esm * esm_proj, batch_dict["protein"], size=num_graphs)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------
        graph_embedding = torch.cat([protein_pool, dna_pool, esm_pool, global_proj], dim=1)
        out = self.linear_transformation(graph_embedding)

        return out


# -----------------------------------------------------------------------------
# Build Model
# -----------------------------------------------------------------------------
def build_model(metadata, hyperparameters: Dict[str, Any], device: torch.device) -> AffinityPredictor:
    """
    Instantiate AffinityPredictor from a configuration dictionary.

    Args:
        metadata (Tuple): The HeteroData metadata (node and edge types).
        hyperparameters (Dict[str, Any]): The configuration used at training time.
        device (torch.device): Device the model is moved to.

    Returns:
        AffinityPredictor: The instantiated model, with its lazy layers still
            unmaterialised.
    """
    model = AffinityPredictor(
                metadata=metadata,
                out_channels=1,
                # Node parameters
                node_hidden_dim=hyperparameters["node_hidden_dim"],
                node_heads=hyperparameters["node_heads"],
                node_proj_dropout=hyperparameters["node_proj_dropout"],
                node_conv_dropout=hyperparameters["node_conv_dropout"],
                # Edge parameters
                edge_dim=hyperparameters["edge_dim"],
                edge_hidden_multiplier=hyperparameters["edge_hidden_multiplier"],
                edge_dropout=hyperparameters["edge_dropout"],
                # ESM parameters
                esm_projected_dim=hyperparameters["esm_projected_dim"],
                esm_hidden_multiplier=hyperparameters["esm_hidden_multiplier"],
                esm_dropout=hyperparameters["esm_dropout"],
                # Global context parameters
                global_projected_dim=hyperparameters["global_projected_dim"],
                global_hidden_multiplier=hyperparameters["global_hidden_multiplier"],
                global_dropout=hyperparameters["global_dropout"],
                # Head parameters
                head_hidden_dim=hyperparameters["head_hidden_dim"],
                head_dropout=hyperparameters["head_dropout"]
            )

    return model.to(device)



# ==============================================================================
# PREDICTION
# ==============================================================================
# Runs the assembled graph through every fold of the trained cross-validation
# ensemble and aggregates their predictions, so the final estimate reflects the
# whole ensemble.

# -----------------------------------------------------------------------------
# Predict Affinity With The Ensemble
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_affinity(graph, folds, hyperparameters, device):
    """
    Predict the binding affinity of one graph with every fold of the ensemble.

    Each fold owns its own (checkpoint, normaliser) pair, since the features
    were scaled with statistics fitted on that fold's training split; the same
    raw graph is therefore re-normalised and re-evaluated.

    Args:
        graph (HeteroData): The raw, unnormalised graph of the complex.
        folds (List[Tuple[int, str, str]]): The (fold, checkpoint_path,
            normalizer_path) triples returned by get_folds.
        hyperparameters (Dict[str, Any]): The architecture configuration used
            at training time, shared by every fold.
        device (torch.device): Device the model and batches are moved to.

    Returns:
        Tuple[float, float]: The mean predicted pKd across folds and its
            standard deviation, the latter measuring the ensemble's
            disagreement rather than a calibrated uncertainty.
    """
    predictions = []
    model = None
    
    for fold, checkpoint_path, normalizer_path in folds:
        normalizers = load_normalizers(normalizer_path)
        normalized_graph = apply_normalizers(graph, normalizers)

        loader = DataLoader([normalized_graph], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)
        
        if model is None:
            model = build_model(batch.metadata(), hyperparameters, device)
            model.eval()
            model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.batch_dict, batch["protein"].esm2, batch.global_context)

        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        
        output = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.batch_dict, batch["protein"].esm2, batch.global_context)
    
        pred_norm = output.detach().cpu().view(-1, 1)
        pred_real = normalizers["target"].inverse_transform(pred_norm).item()
        
        predictions.append(pred_real)
        print(f"Fold {fold}: pKd = {pred_real:.3f}")
        
    return float(np.mean(predictions)), float(np.std(predictions))


# -----------------------------------------------------------------------------
# Locate The Trained Ensemble
# -----------------------------------------------------------------------------
def get_folds(gnn_path):
    """
    Pair up every fold's checkpoint with its matching normaliser bundle. Both
    files are expected to encode the same fold number in their name.

    Args:
        gnn_path (str): Root directory of the trained ensemble, expected to
            contain a "folds" subdirectory (model checkpoints) and a
            "normalizers" subdirectory (fitted normaliser bundles).

    Returns:
        List[Tuple[int, str, str]]: The (fold, checkpoint_path,
            normalizer_path) triples, sorted by fold number.
    """
    fold_dir = os.path.join(gnn_path, "folds")
    normalizer_dir = os.path.join(gnn_path, "normalizers")

    fold_num_pattern = re.compile(r"fold(\d+)", re.IGNORECASE)

    def extract_fold(path):
        match = fold_num_pattern.search(os.path.basename(path))
        if not match:
            raise ValueError(f"Cannot extract the fold number from: {path}")
        return int(match.group(1))

    checkpoints = {extract_fold(path): path for path in list_files(fold_dir, "*.pt")}
    normalizers = {extract_fold(path): path for path in list_files(normalizer_dir, "*.pt")}

    paired = sorted(set(checkpoints) & set(normalizers))

    if not paired:
        raise SystemExit(f"[ERROR] No (checkpoint, normaliser) pair found under {gnn_path}")

    return [(fold, checkpoints[fold], normalizers[fold]) for fold in paired]



# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
# Orchestrates the run for a single complex: build its graph from the two
# input files, run it through every fold of the trained ensemble, and report
# the mean predicted pKd and its dispersion across folds.

# -----------------------------------------------------------------------------
# Main Workflow
# -----------------------------------------------------------------------------
def main(json_path: str, esm2_path: str, model_dir: str) -> None:
    """
    Predict the binding affinity of a single complex.

    The complex is assembled into a graph from its standardized JSON and its
    ESM-2 embeddings, then evaluated with every (checkpoint, normaliser) pair
    found under "model_dir", which is the trained cross-validation ensemble,
    it is not specific to the complex being predicted. The final estimate is
    the mean pKd across folds, with its standard deviation reported as a
    measure of the ensemble's disagreement.

    Args:
        json_path (str): Path to the standardized JSON file of the complex.
        esm2_path (str): Path to the ESM-2 .pt file of the complex.
        model_dir (str): Root directory holding the "folds" and "normalizers"
            subdirectories of the trained ensemble.
    """
    print("\n# ============================================== #")
    print("#  STARTING AFFINITY PREDICTION (SINGLE COMPLEX)  #")
    print("# ============================================== #\n")

    print(f"[INFO] Input JSON path: {json_path}")
    print(f"[INFO] Input ESM-2 path: {esm2_path}")
    print(f"[INFO] Trained model directory: {model_dir}")

    # ------------------------------------------
    # Locate the trained ensemble
    # ------------------------------------------
    folds = get_folds(model_dir)
    print(f"[INFO] Ensemble folds found: {len(folds)}")

    # ------------------------------------------
    # Build the graph
    # ------------------------------------------
    print("\n[1/2] Building the heterogeneous graph...")
    encoder = generate_encoders()
    graph = build_graph(json_path, esm2_path, encoder)
    print(f"[INFO] Structure identified")

    # ------------------------------------------
    # Predict
    # ------------------------------------------
    print(f"\n[2/2] Predicting binding affinity...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Hardware accelerator: {device.type.upper()}")

    mean_pkd, sd_pkd = predict_affinity(graph, folds, HYPERPARAMETERS, device)

    print(f"\n[SUCCESS] Predicted pKd: {mean_pkd:.3f} ± {sd_pkd:.3f}")

    print("\n# ================================ #")
    print("#  AFFINITY PREDICTION COMPLETED!  #")
    print("# ================================ #\n")


# -----------------------------------------------------------------------------
# CLI Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Configure command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Single-complex binding-affinity predictor from a standardized JSON file and its ESM-2 embeddings.",
        epilog=f"Developed by {__author__}. Version {__version__}.",
    )

    parser.add_argument("-j", "--json", type=str, required=True,
                        help="Path to the standardized JSON file of the complex.")
    parser.add_argument("-e", "--esm2", type=str, required=True,
                        help="Path to the ESM-2 .pt file of the complex.")
    parser.add_argument("-m", "--model", required=True,
                        help="Root directory of the trained model ('folds' and 'normalizers' subdirectories).")

    args = parser.parse_args()

    main(args.json, args.esm2, args.model)
