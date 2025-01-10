from rdkit import Chem
from rdkit.Chem import AllChem

# SMILES string
smiles_string = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"  # this is for aspirin

# Convert SMILES to mol
mol = Chem.MolFromSmiles(smiles_string)

# Add Hydrogens
mol = Chem.AddHs(mol)

# Generate 3D coordinates
AllChem.EmbedMolecule(mol)

# Write to PDB file
Chem.MolToPDBFile(mol, "output.pdb")