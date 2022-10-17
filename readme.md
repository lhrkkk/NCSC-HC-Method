# NCSC-HC Peptide conformational ensemble evaluation + clustering method

## Function
Read peptide dihedral angle data in .csv or .xls format.
The normalized conformational spatial coverage of the ensemble and (1-Gini coefficient) are calculated and output, the former representing the diversity of the conformational ensemble and the latter representing the uniformity of distribution. The values for both are positively correlated with the quality of the conformational ensemble.
Plot the energy "state density map" of the ensemble.
Cluster conformational ensembles, output clustering results in .csv format, and visualize clustering results.

## Environment dependence

### Python version
    python 3

### Third-party libraries
    numpy 1.19.2
    sklearn 0.24.0
    matplotlib 3.3.3

## Usage
    Command line call
    ```
    python eval_and_cluster.py [-h] [-n N] [-s S] [-e E] [-i <filename>] [-o <filename>] [--cluster] [--sidechain] [--show] [--stg]
    ```

## Parameter interpretation

    -h View help
    -n The number of amino acids contained in the current polypeptide conformational ensemble.
    -s The amino acid sequence of the current peptide, which is represented by a single capital letter abbreviation. For example, "glycine-alanine" dipeptide abbreviated as "VA".
    -e Conformational energy truncation. Conformations with energies above this value will not be read and will not participate in subsequent calculations.
    -i .csv or .xls format of the peptide dihedral data file path.
    -o Clustering result output file path. The default is the current directory. The file name is the input file followed by "_clusteropt".
    --cluster Whether to perform conformational clustering algorithms. The default is not to proceed.
    --sidechain Clustering algorithm whether to consider sidechain dihedral angles. The default is not considered.
    --show  Whether to visualize the clustering results. The default is not to proceed.
    --stg Whether to draw a state density map, the default is not to draw.

## Enter the file grid

### XLS lattice

| Conformational file name | Conformational dihedral angles (°) separated by spaces | Conformational energy (kcal/mol) |
|----------:|------------------------:|-----------------:|
|  12345.xyz|   -180.0 45.0 90.0 135.0|             4.321|
|  54321.xyz|   120.0 120.0 90.0 -45.0|             1.234|

### csv lattice
    Conformation file name, space-separated conformational dihedral angle (°), conformational energy (kcal/mol).
    12345.xyz,-180.0 45.0 90.0 135.0,4.321
    54321.xyz,120.0 120.0 90.0 -45.0,1.234

# NCSC-HC-Method
