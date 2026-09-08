Protein-DNA Binding Affinity Prediction Project
==============================
### A Heterogeneous Graph Neural Network to Predict Protein-DNA Binding Affinity from Complex 3D Structure.

### Data
The data utilized in this project was compiled by cross-referencing three public databases:

* **[Protein Data Bank (PDB) v3.2](https://www.rcsb.org/)**: Source of experimentally determined 3D structures and atomic coordinates.
* **[DNAproDB v2.0](https://dnaprodb.usc.edu/)**: Structural and physicochemical characterization of protein-DNA complexes.
* **[PDBbind+ v2020.R1](http://www.pdbbind.org.cn/)**: Experimentally measured binding affinities.

Additional tools used to generate features:
* **[PDB2PQR](https://pdb2pqr.readthedocs.io/)**: Computes atomic partial charges and Van der Waals radii via automated protonation-state assignment.
* **[ESM-2](https://github.com/facebookresearch/esm)**: Protein language model used to generate evolutionary per-residue embeddings.

### Important libraries and tools used
* **PyTorch / PyTorch Geometric**: For building and training the heterogeneous graph neural network
* **ESM-2 (fair-esm)**: For generating evolutionary protein embeddings
* **PDB2PQR**: For computing atomic partial charges and Van der Waals radii
* **Biopython**: For parsing and handling PDB structures
* Other common data science packages were used, such as numpy, pandas, scikit-learn, and matplotlib

### Acknowledgement
I would like to thank my supervisor José Ramón Valverde from the Scientific Computing group at the National Center for Biotechnology (CNB-CSIC), for his support throughout this project.

## Abstract
Protein–DNA recognition is essential for the regulation of gene expression, genome replication, and genome repair, and its thermodynamic quantification is indispensable for the design of artificial transcription factors and the development of targeted therapies. Experimental determination of these interactions, however, is costly and has low throughput, which has driven the development of computational predictive methods. Nevertheless, computational approaches have significant methodological limitations that restrict their accuracy and generalizability.

This project presents a heterogeneous graph neural network model for the quantitative prediction of protein–DNA binding affinity based on the three-dimensional structure of the complex. By integrating the Protein Data Bank, DNAproDB, and PDBbind+ databases, a dataset of 491 complexes was constructed, which is larger than those used by previous methods. Each complex is represented as a graph with two types of nodes, nucleotides and amino acids, and ten relationships that preserve the physical nature of each interaction, incorporating physicochemical, geometric, and electrostatic descriptors along with evolutionary representations derived from the ESM-2 language model.

Through five-fold cross-validation, the model achieves a Pearson correlation of r = 0.63 ± 0.06 and a mean absolute error of 1.37 ± 0.10 kcal/mol. This performance is achieved using a single generalist architecture, far surpassing the values of 0.12 to 0.21 obtained by previous methods without subclassification. Interpretability analysis reveals that the prediction relies predominantly on evolutionary information, while the physical attributes of the edges or nodes remain underutilized.

<img src="https://github.com/msanchezoliveros/protein-dna-binding-affinity-gnn/blob/main/figures/F1_heterogeneous_gnn_affinity_prediction_model.png?raw=true" alt="Heterogeneous GNN affinity prediction model" width="700"/>

# Background Information

### Graph construction
Each complex is represented as a heterogeneous graph with two types of nodes and ten relationships (five types + their inverses):

- **DNA nodes** (20-dimensional vector): chemical identity (one-hot of canonical and modified bases), conformational state (secondary structure, glycosidic conformation), solvent exposure (FASA), degree of interface involvement, and charge (PDB2PQR).
- **Protein nodes** (35-dimensional vector): amino acid identity (one-hot), charge, interface involvement, extended solvent exposure (backbone and side-chain FASA/SESA), helical coordinates, and local surface geometry (curvature, SAP score).
- **DNA base-pairing edges** (13-dimensional vector): Pair type (Watson-Crick / Hoogsteen / other, one-hot), mismatch flag, hydrogen-bond distance statistics (count, minimum, mean), and six local shape parameters (buckle, propeller, opening, shear, stagger, stretch).
- **DNA stacking edges** (7-dimensional vector): Minimum inter-base distance and six base-step shape parameters (twist, roll, rise, slide, tilt, shift).
- **DNA backbone edges** (1-dimensional vector): Phosphodiester linkage distance between consecutive nucleotides.
- **Protein backbone edge** (1-dimensional vector): Peptide-bond linkage gap between consecutive residues.
- **Interface edge** (46-dimensional vector): Connects each amino acid to the nucleotides it contacts — contact geometry (three distances plus a one-hot geometry type), buried surface area (BASA) split by moiety, hydrogen bonds (direct and water-mediated, per-moiety counts plus distance statistics), and Van der Waals contacts (per-moiety counts plus distance statistics).
- **Global context** (96-dimensional vector): Experimental conditions (resolution, temperature, pH), per-chain interface descriptors (interaction counts, hydrophobicity, surface curvature ratios, amino-acid recognition propensities), and DNA descriptors (entity topology and modifications, helical-segment classification, geometry, sequence composition, and structural motifs).

### Model architecture
The model consists of four independent encoders (nodes, edges, ESM-2, and global context), combined through an attention mechanism and concatenated before the final regression head:

1. **Node encoder**: Linear projection → LayerNorm → GELU → Dropout (dh = 24 channels).
2. **Message passing**: **GATv2** convolution with 4 averaged attention heads, conditioned on edge attributes, followed by a **GRU** cell that integrates the refined message with the node's previous state (reduces oversmoothing).
3. **Edge encoder**: Two-layer perceptron per relationship type, projected to a shared width (de = 16 channels).
4. **ESM-2 encoder**: Projection of the evolutionary embeddings (1280 → 16 channels), controlled integration during training.
5. **Global context encoder**: Two-layer perceptron (96 → 8 channels).
6. **Regression head**: Concatenation of the four resulting vectors (72 dimensions) → LayerNorm → GELU → Dropout → linear projection to a single scalar value (pKd).

### Training
- **Cross-validation**: 5-fold scheme, stratified by quantiles of experimental pKd. Per fold: 313 training graphs, 79 validation graphs, 99 test graphs.
- **Normalization**: Per-column standardization (robust scaling if |skewness| > 1.0, standard scaling otherwise), fit exclusively on the training subset of each fold.
- **Loss function**: Huber loss (δ = 0.75) on normalized values.
- **Optimizer**: AdamW (initial lr = 1×10⁻⁴, weight decay = 5×10⁻²), batch size = 32 graphs.
- **Scheduling**: Learning rate halved after 10 epochs without improvement (minimum 1×10⁻⁶); early stopping after 25 epochs without improvement; maximum 500 epochs.
- **Hardware**: NVIDIA GeForce RTX 5070 Mobile.

### Evaluation metrics
- **R²**: Coefficient of determination.
- **Pearson correlation (r)**: Linear correlation.
- **MAE**: Mean absolute error, converted to kcal/mol via ΔG = -RT ln(10) · pKd.

## Results
Evaluated via 5-fold cross-validation, the model achieved the following performance on the test sets:

| Metric | Value (mean ± SD) |
|---|---|
| R² | 0.38 ± 0.10 |
| Pearson correlation (r) | 0.63 ± 0.06 |
| MAE | 1.37 ± 0.10 kcal/mol |

<img src="https://github.com/msanchezoliveros/protein-dna-binding-affinity-gnn/blob/main/figures/F8_scatter_predictions.png?raw=true" alt="Predicted vs experimental affinity" width="500"/>

This performance was achieved using a single generalist architecture, trained on the entire dataset without subclassification, exceeding the correlation values (0.12–0.21) obtained by comparable methods when trained without subclassification. Interpretability analysis (block-neutralization ablation and integrated gradients) showed that the prediction relies predominantly on the ESM-2 evolutionary representation, while the physical attributes of the edges remain underutilized.

<img src="https://github.com/msanchezoliveros/protein-dna-binding-affinity-gnn/blob/main/figures/F9_ablation_study.png?raw=true" alt="Block neutralization ablation" width="500"/>

## Usage

The steps below describe how to predict the binding affinity of a single protein-DNA complex, starting from its raw structure file. Placeholders are written in angle brackets (`<complex_id>`), the standard documentation convention for a value the user must substitute with their own file or structure name — none of the steps depend on a specific, hardcoded example.

### 1. Generate the structural JSON with DNAproDB

The complex's 3-D structure (mmCIF or PDB format) is first processed with the [DNAproDB pipeline](https://github.com/timkartar/dnaprodb), which extracts the structural, geometric, and interaction descriptors used throughout this project and outputs a single `<complex_id>.json` file. Clone the DNAproDB repository and follow its own installation and execution instructions:
 
```bash
git clone https://github.com/timkartar/dnaprodb.git
```

The resulting `<complex_file>.json` file is the required input for the next steps.

### 2. Compute electrostatic properties (PDB2PQR)

Partial atomic charges and Van der Waals radii are computed with [PDB2PQR](https://pdb2pqr.readthedocs.io/), using the AMBER force field and PROPKA to assign protonation states at the experimental pH of the structure:

```bash
pdb2pqr --ff=AMBER \
    --titration-state-method=propka \
    --with-ph="<pH>" \
    --keep-chain \
    --whitespace \
    --drop-water \
    --apbs-input "<complex_file>.in" \
    "<complex_file>.pdb" "<complex_file>.pqr"
```

* `<complex_file>.pdb` — input structure of the complex (PDB format).
* `<pH>` — experimental pH of the structure.

This saves the per-atom partial charges and Van der Waals radii to `<complex_file>.pqr`, and automatically generates the APBS configuration file `<complex_file>.in`.

### 3. Compute ESM-2 embeddings

Per-residue evolutionary embeddings are then generated from the protein sequence(s) of the complex using the ESM-2 language model:

```bash
python3 src/features/compute_esm2.py \
    --json "<complex_file>.json" \
    --esm "esm2_t33_650M_UR50D" \
    --half \
    --output "<complex_file>.pt"
```

* `<complex_file>.json` — the raw DNAproDB export generated in step 1.
* `--esm` — optional argument specifying the ESM-2 model version (defaults to `esm2_t33_650M_UR50D` if omitted).

This saves the per-residue embedding of every amino acid in the complex to `<complex_file>.pt`.

### 4. Build the standardized JSON

The raw DNAproDB entry from step 1 and the electrostatics from step 2 are then combined, together with the experimental conditions of the structure, into the standardized JSON consumed by the prediction step:

```bash
python3 src/features/build_models.py \
    --json "<complex_file>.json" \
    --pqr "<complex_file>.pqr" \
    --resolution "<complex_resolution>" \
    --temperature "<complex_temperature>" \
    --ph "<complex_ph>" \
    --output "<complex_file>_standardized.json"
```

* `<complex_file>.json` — the raw DNAproDB export generated in step 1.
* `<complex_file>.pqr` — the computed electrostatics generated in step 2.
* `<complex_resolution>` — experimental resolution of the structure (Å).
* `<complex_temperature>` — experimental temperature of the structure (K).
* `<complex_ph>` — experimental pH of the structure, the same value used in step 2's `--with-ph`.

This saves the standardized model formatted for the graph neural network to `<complex_file>_standardized.json`.

### 5. Build the graph and predict the binding affinity

The complex is finally assembled into its heterogeneous graph and evaluated with the trained cross-validation ensemble:

```bash
python3 src/features/predict_affinity.py \
    --json "<complex_id>_standardized.json" \
    --esm2 "<complex_id>.pt" \
    --model-dir "<gnn_model_dir>"
```

* `<complex_id>_standardized.json` — the standardized JSON built in step 4.
* `<complex_id>.pt` — the ESM-2 embeddings produced in step 3.
* `<gnn_model_dir>` — the trained ensemble directory, containing a `folds/` subdirectory (one checkpoint per cross-validation fold) and a `normalizers/` subdirectory (the matching fitted normalizers), available for download from this repository.

The script prints one line per fold with its individual prediction, followed by the final ensemble estimate.
The reported value is the mean predicted pKd across the five folds, and its standard deviation reflects the ensemble's internal disagreement rather than a calibrated confidence interval.
