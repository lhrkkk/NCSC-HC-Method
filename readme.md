# NCSC-HC Peptide conformational ensemble evaluation + clustering method

## Function
Read peptide dihedral angle data in .csv or .xls format.

The normalized conformational spatial coverage of the ensemble and (1-Gini coefficient) are calculated and outputed.

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
    python eval_and_cluster.py [-h] [-n N] [-s S] [-e E] [-ncfm N] [-i <filename>] [-o <filename>] [--cluster] [-clustermethod S] [--sidechain] [--split] [-splitthreshold DEG] [--show] [--stg]
    ```

## Parameter interpretation

    -h View help
    -n The number of amino acids contained in the current polypeptide conformational ensemble.
    -s The amino acid sequence of the current peptide, which is represented by a single capital letter abbreviation. For example, "glycine-alanine" dipeptide abbreviated as "VA".
    -e Conformational energy truncation. Conformations with energies above this value will not be read and will not participate in subsequent calculations.
    -ncfm   number truncation, that is, a fixed amount of data is selected by energy from low to high. When energy truncation is set, if the resulting ensemble size is less than this parameter, it will take no effect. The default is infinity.
    -i .csv or .xls format of the peptide dihedral data file path.
    -o Clustering result output file path. The default is the current directory. The file name is the input file followed by "_clusteropt".
    --cluster Whether to perform conformational clustering algorithms. The default is not to proceed.
    -clustermethod uses a conformational clustering algorithm with two options, "birch" and "kmeans", which defaults to "birch"
    --sidechain Clustering algorithm whether to consider sidechain dihedral angles. The default is not considered.
    --split         Whether to divide the two-sided angle of the main chain to correct the clustering result, by default.
    -splitthreshold The division threshold used by the correction algorithm, 90° by default.
    --show  Whether to visualize the clustering results. The default is not to proceed.
    --stg Whether to draw a state density map, the default is not to draw.



## Input file

### XLS format

| Conformational file name | Conformational dihedral angles (°) separated by spaces | Conformational energy (kcal/mol) |
|----------:|------------------------:|-----------------:|
|  12345.xyz|   -180.0 45.0 90.0 135.0|             4.321|
|  54321.xyz|   120.0 120.0 90.0 -45.0|             1.234|

### csv format
    Conformation file name, space-separated conformational dihedral angle (°), conformational energy (kcal/mol).
    12345.xyz,-180.0 45.0 90.0 135.0,4.321
    54321.xyz,120.0 120.0 90.0 -45.0,1.234

# Contributors:
Fangning Ren, fangning.ren@emory.edu

Haorui Lu, lhrkkk@mail.ustc.edu.cn

