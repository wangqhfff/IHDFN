System Requirements

Python 3.8.10

RDKit 2022.09.5

torch: 1.11.0+cu113

dgl: 0.9.1.post1

dgllife: 0.3.2

numpy: 1.23.5

scikit-learn: 1.1.3

pandas: 1.5.3  



Datasets

All the datasets we use are publicly available.The 'datasets' folder of this project already contains all the required datasets.

BioSNAP and initial version DAVIS datasets can be obtained at https://github.com/kexinhuang12345/MolTrans;

Human source is at https://github.com/lifanchen-simm/transformerCPI.



Run IHDFN on Our Data to Reproduce Results

python main.py --cfg "configs/IHDFN.yaml" --data "biosnap" --split "random"

python main.py --cfg "configs/IHDFN.yaml" --data "human" --split "cold"

python main.py --cfg "configs/IHDFN.yaml" --data "human" --split "random"

python main.py --cfg "configs/IHDFN.yaml" --data "DAVIS" --split "random"

Interpretable

In the 'interpretation-2.py' file, we conduct molecule-level interpretability analysis. The file 'interpretation-sample.csv' holds a case study from our paper.To reproduce the images mentioned in our paper, simply run the file after modifying the path as needed.

Dear editors and reviewers, please feel free to contact us if there are any issues during the software execution！
