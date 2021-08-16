# Graph neural network for predicting NMR chemical shifts
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ur-whitelab/nmrgnn/blob/master/colab/NMRPredictor.ipynb)

This library is the code and a pre-trained model to predict NMR chemical shifts from protein structures and organic molecules. It relies on the [nmrdata](https://github.com/ur-whitelab/nmrdata) package which includes embeddings and NMR parameters.

## Install

Install using pip

```sh
pip install nmrgnn
```

## Colab

To use this package without installing, [use this colab](https://colab.research.google.com/github/ur-whitelab/nmrgnn/blob/master/colab/NMRPredictor.ipynb)

## Command Line Usage

Available commands are

* `nmrgnn eval-struct` to predict chemical shifts of structure via MDAnalysis library as coordinate reader
* `nmrgnn train` to train a model
* `nmrgnn hyper` to tune hyperparameters
* `nmrgnn eval-tfrecords` to evaluate model on records in format from `nmrdata` package

### Predict NMR Chemical Shfits

*Note: This model is trained on models with no solvent, so remove that before use. For small molecules, the model was trained mostly on water solutions. You should
only expect agreement in relative chemical shifts between atoms depending on your solvent and reference.*

To predict NMR chemical shifts via the MDAnalysis library as a reader:

```sh
nmrgnn eval-struct [struct-file] [output-csv]
```

where `struct-file` could be a pdb file or equivalent. Example:

```
nmrgnn eval-struct 108M.pdb 108M-predicted.csv
```

## Warning about Peaks

If you receive a warning about peaks being poor, you likely have no hydrogens in your protein. You can add using online tools or use these commands to fix quickly by
using [OpenMM](https://openmm.org/)

```sh
conda install -y -c omnia openmm
pip install nmrdata[parse]@git+git://github.com/ur-whitelab/nmrgnn.git
nmrparse clean-pdb [your-pdb] [your-pdb]-H.pdb
```

## Library Usage

Available functions are

* `load_model` to load the included pre-trained model or specify a path to a trained model
* `universe2graph` to convert an MDAnalysis universe into a tuple of atoms, neighbor list, edges, inverse_degree.
* `check_peaks` to estimate validity of predicted peaks

The example below predicts peaks and estimates (`True/False`) if the peaks are valid. Examples of why peaks are
not valid are that the elements are not inlcuded in training data (e.g., oxygen shifts) or unusual chemistries or
you forgot to remove solvent.

```py
import MDAnalysis as md
import nmrgnn

model = nmrgnn.load_model()
u = md.Universe('108M.pdb')
g = nmrgnn.universe2graph(u)
peaks = model(g)
# check_peaks only uses first element of tuple (atom identities)
confident = nmrgnn.check_peaks(g[0], peaks)
```

**You should not trust peaks coming from model without checking**

### Analyzing Trajectories

Here is an example for analzying a trajectory

```py
import MDAnalysis as md
import nmrgnn

model = nmrgnn.load_model()

u = md.Universe(PATH_TO_FILES)
for ts in u.trajectory:
    x = nmrgnn.universe2graph(u)
    peaks = model(x)
    nmrgnn.check_peaks(x[0], peaks)    
    # do something with peaks
    ....
```

## Citation

Please cite [Predicting Chemical Shifts with Graph Neural Networks](https://pubs.rsc.org/en/content/articlehtml/2021/sc/d1sc01895g)

```bibtex
@article{yang2021predicting,
  title={Predicting Chemical Shifts with Graph Neural Networks},
  author={Yang, Ziyue and Chakraborty, Maghesree and White, Andrew D},
  journal={Chemical Science},
  year={2021},
  publisher={Royal Society of Chemistry}
}
```
## Model Performance

Here is the included model performance on proteins (`P` prefix) and organic molecules (`Mol` prefix). `r` is correlation coefficient and `rmsd` is root mean square deviation. These results vary from paper values because they are evaluated on whole proteins instead of 256 atom fragments.

|             |    N | baseline            |
| :---------- | ---: | :------------------ |
| Mol-H-r     |  307 | 0.9591749434360993  |
| Mol-H-rmsd  |  307 | 0.39710393617916234 |
| P-C-r       | 6701 | 0.864163            |
| P-H-r       | 7747 | 0.72265             |
| P-N-r       | 7640 | 0.890842            |
| P-CA-r      | 8305 | 0.97374             |
| P-CB-r      | 6827 | 0.990706            |
| P-CD-r      |  739 | 0.996123            |
| P-CD1-r     |  961 | 0.999515            |
| P-CD2-r     |  609 | 0.999223            |
| P-CE-r      |  340 | 0.991736            |
| P-CE1-r     |  261 | 0.958121            |
| P-CE2-r     |  173 | 0.943739            |
| P-CE3-r     |   37 | -0.215088           |
| P-CG-r      | 1674 | 0.998763            |
| P-CG1-r     |  589 | 0.93124             |
| P-CG2-r     |  839 | 0.829016            |
| P-CH2-r     |   43 | 0.158363            |
| P-CZ-r      |  125 | 0.984575            |
| P-CZ2-r     |   45 | 0.311805            |
| P-CZ3-r     |   37 | 0.164961            |
| P-HA-r      | 5565 | 0.839377            |
| P-HA2-r     |  462 | 0.495514            |
| P-HA3-r     |  449 | 0.262298            |
| P-HB-r      |  960 | 0.958713            |
| P-HB2-r     | 3427 | 0.901358            |
| P-HB3-r     | 3255 | 0.901234            |
| P-HD1-r     |  383 | 0.44733             |
| P-HD11-r    |  753 | 0.615756            |
| P-HD12-r    |  753 | 0.585852            |
| P-HD13-r    |  753 | 0.609181            |
| P-HD2-r     | 1043 | 0.988991            |
| P-HD21-r    |  428 | 0.617599            |
| P-HD22-r    |  428 | 0.651927            |
| P-HD23-r    |  428 | 0.605888            |
| P-HD3-r     |  637 | 0.95089             |
| P-HE-r      |   93 | 0.396258            |
| P-HE1-r     |  413 | 0.879142            |
| P-HE2-r     |  561 | 0.98963             |
| P-HE3-r     |  293 | 0.985685            |
| P-HG-r      |  389 | 0.810401            |
| P-HG1-r     |   11 | 0.0653286           |
| P-HG11-r    |  350 | 0.572609            |
| P-HG12-r    |  350 | 0.498696            |
| P-HG13-r    |  350 | 0.558426            |
| P-HG2-r     | 1317 | 0.867619            |
| P-HG21-r    |  936 | 0.689592            |
| P-HG22-r    |  936 | 0.674086            |
| P-HG23-r    |  936 | 0.662057            |
| P-HG3-r     | 1200 | 0.856177            |
| P-HH-r      |    1 | nan                 |
| P-HH2-r     |   51 | 0.217372            |
| P-HZ-r      |  134 | 0.407285            |
| P-HZ2-r     |   54 | 0.419415            |
| P-HZ3-r     |   45 | 0.318577            |
| P-ND1-r     |    9 | 0.184443            |
| P-ND2-r     |  173 | 0.320299            |
| P-NE-r      |   88 | 0.0135033           |
| P-NE1-r     |   64 | 0.0998792           |
| P-NE2-r     |  149 | 0.972614            |
| P-NH1-r     |    3 | -0.914066           |
| P-NH2-r     |    3 | -0.276087           |
| P-NZ-r      |    1 | nan                 |
| P-C-rmsd    | 6701 | 1.22819             |
| P-H-rmsd    | 7747 | 0.279766            |
| P-N-rmsd    | 7640 | 6.65505             |
| P-CA-rmsd   | 8305 | 1.3298              |
| P-CB-rmsd   | 6827 | 3.10571             |
| P-CD-rmsd   |  739 | 10.3192             |
| P-CD1-rmsd  |  961 | 2.74597             |
| P-CD2-rmsd  |  609 | 4.35399             |
| P-CE-rmsd   |  340 | 1.14623             |
| P-CE1-rmsd  |  261 | 4.69154             |
| P-CE2-rmsd  |  173 | 4.82229             |
| P-CE3-rmsd  |   37 | 3.0327              |
| P-CG-rmsd   | 1674 | 1.63828             |
| P-CG1-rmsd  |  589 | 1.558               |
| P-CG2-rmsd  |  839 | 1.87753             |
| P-CH2-rmsd  |   43 | 1.95861             |
| P-CZ-rmsd   |  125 | 4.32496             |
| P-CZ2-rmsd  |   45 | 1.22984             |
| P-CZ3-rmsd  |   37 | 1.99567             |
| P-HA-rmsd   | 5565 | 0.0903255           |
| P-HA2-rmsd  |  462 | 0.119584            |
| P-HA3-rmsd  |  449 | 0.234069            |
| P-HB-rmsd   |  960 | 0.103812            |
| P-HB2-rmsd  | 3427 | 0.10552             |
| P-HB3-rmsd  | 3255 | 0.117287            |
| P-HD1-rmsd  |  383 | 0.114696            |
| P-HD11-rmsd |  753 | 0.0699893           |
| P-HD12-rmsd |  753 | 0.0744762           |
| P-HD13-rmsd |  753 | 0.0711484           |
| P-HD2-rmsd  | 1043 | 0.105893            |
| P-HD21-rmsd |  428 | 0.0737762           |
| P-HD22-rmsd |  428 | 0.0689306           |
| P-HD23-rmsd |  428 | 0.0764191           |
| P-HD3-rmsd  |  637 | 0.0869007           |
| P-HE-rmsd   |   93 | 0.422132            |
| P-HE1-rmsd  |  413 | 0.376196            |
| P-HE2-rmsd  |  561 | 0.0861489           |
| P-HE3-rmsd  |  293 | 0.0855213           |
| P-HG-rmsd   |  389 | 0.118694            |
| P-HG1-rmsd  |   11 | 10.3704             |
| P-HG11-rmsd |  350 | 0.0504736           |
| P-HG12-rmsd |  350 | 0.0552385           |
| P-HG13-rmsd |  350 | 0.0516929           |
| P-HG2-rmsd  | 1317 | 0.0654069           |
| P-HG21-rmsd |  936 | 0.0634577           |
| P-HG22-rmsd |  936 | 0.0650697           |
| P-HG23-rmsd |  936 | 0.0679991           |
| P-HG3-rmsd  | 1200 | 0.0775636           |
| P-HH-rmsd   |    1 | 4.07231             |
| P-HH2-rmsd  |   51 | 0.0862706           |
| P-HZ-rmsd   |  134 | 0.147387            |
| P-HZ2-rmsd  |   54 | 0.13507             |
| P-HZ3-rmsd  |   45 | 0.083249            |
| P-ND1-rmsd  |    9 | 1576.13             |
| P-ND2-rmsd  |  173 | 6.56618             |
| P-NE-rmsd   |   88 | 231.589             |
| P-NE1-rmsd  |   64 | 4.51713             |
| P-NE2-rmsd  |  149 | 13.9975             |
| P-NH1-rmsd  |    3 | 5.76985             |
| P-NH2-rmsd  |    3 | 0.91028             |
| P-NZ-rmsd   |    1 | 165.069             |
