from collections import defaultdict
import pickle
import numpy as np
import requests

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

atom_dict = load_pickle('app/trained_weights/atom_dict.pickle')
bond_dict = load_pickle('app/trained_weights/bond_dict.pickle')
fingerprint_dict = load_pickle('app/trained_weights/fingerprint_dict.pickle')
radius = 2

# dictionary of atoms where a new element gets a new index
def create_atom_index(mol):
    atoms = [atom_dict[a.GetSymbol()] for a in mol.GetAtoms()]
    atoms = np.array(atoms)
    return atoms

# format from_atomIDx : [to_atomIDx, bondDict]
def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def create_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        vertices = atoms
        for _ in range(radius):
            fingerprints = []
            for i, j_bond in i_jbond_dict.items():
                neighbors = [(vertices[j], bond) for j, bond in j_bond]
                fingerprint = (vertices[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            vertices = fingerprints

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    n = adjacency.shape[0]
    adjacency = adjacency + np.eye(n)
    degree = sum(adjacency)
    d_half = np.sqrt(np.diag(degree))
    d_half_inv = np.linalg.inv(d_half)
    adjacency = np.matmul(d_half_inv,np.matmul(adjacency,d_half_inv))
    return np.array(adjacency)

def smiles_to_iupac(smiles):
    rep = "iupac_name"
    CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def preprocess(mol, name):
    file_name = "app/static/images/molecule"+name+".png"
    f = open("app/static/images/molecule"+name+"111.png", 'w')
    Draw.MolToFile(mol, file_name, size= (250, 250), kekulize=True, wedgeBonds=True)
    f.close()

    atoms = create_atom_index(mol)
    i_jbond_dict = create_ijbonddict(mol)
    Fingerprints = create_fingerprints(atoms, i_jbond_dict, radius)
    Adjacency = create_adjacency(mol)

    desc = np.zeros((7,))
    desc[0] = Descriptors.MolMR(mol)
    desc[1] = Descriptors.MolLogP(mol)
    desc[2] = Descriptors.MolWt(mol)
    desc[3] = Descriptors.NumRotatableBonds(mol)
    desc[4] = Descriptors.NumAliphaticRings(mol)
    desc[5] = Descriptors.NumAromaticRings(mol)
    desc[6] = Descriptors.NumSaturatedRings(mol)

    return Fingerprints, Adjacency, desc

class PathwayPredictor(nn.Module):
    def __init__(self):
        super(PathwayPredictor, self).__init__()
        self.embed_atom = nn.Embedding(n_fingerprint, dim)
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
        self.W_property = nn.Linear(dim+extra_dim, 11)
    
    """Pad adjacency matrices for batch processing."""
    def pad(self, matrices, value):
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            pad_matrices[m:m+s_i, m:m+s_i] = d
            m += s_i
        return torch.FloatTensor(pad_matrices).to(device)
    
    def sum_axis(self, xs, axis):
        y = list(map(lambda x: torch.sum(x, 0), torch.split(xs, axis)))
        return torch.stack(y)
    
    def update(self, xs, adjacency, i):
        hs = torch.relu(self.W_atom[i](xs))
        return torch.matmul(adjacency, hs)
    
    def forward(self, inputs, sel_desc):
        atoms, adjacency = inputs
        axis = list(map(lambda x: len(x), atoms))
        atoms = torch.cat(atoms)
        x_atoms = self.embed_atom(atoms)
        adjacency = self.pad(adjacency, 0)

        for i in range(layer):
            x_atoms = self.update(x_atoms, adjacency, i)
        
        extra_inputs = sel_desc.to(device)
        y_molecules = self.sum_axis(x_atoms, axis)
        y_molecules = torch.cat((y_molecules,extra_inputs),1)
        z_properties = self.W_property(y_molecules)
        
        return z_properties

dim = 70
extra_dim = 7
layer = 2
n_fingerprint = len(fingerprint_dict)

model = PathwayPredictor().to(device)
model.load_state_dict(torch.load("app/trained_weights/gnc.pth"))

def predict(smiles):
    mol = Chem.MolFromSmiles(smiles['text'].strip())
    if not mol:
        return {'class': 0}

    name = smiles_to_iupac(smiles['text'].strip())
    Fingerprints, Adjacency, descriptors = preprocess(mol, name)
    
    Fingerprints = torch.LongTensor(Fingerprints).to(device)
    inputs = list(zip([Fingerprints, Adjacency]))
    sel_desc = torch.FloatTensor(np.array([descriptors])) 
    
    z_properties = model.forward(inputs, sel_desc)
    p_properties = torch.sigmoid(z_properties)
    p_properties = p_properties.data.to(device).numpy()

    p_properties[p_properties<0.5]  = 0
    p_properties[p_properties>=0.5] = 1

    classes = ['Carbohydrate metabolism', 'Energy metabolism', 'Lipid metabolism', 'Nucleotide metabolism', 'Amino acid metabolism', 
                'Metabolism of other amino acids', 'Glycan biosynthesis and metabolism', 'Metabolism of cofactors and vitamins', 
                'Metabolism of terpenoids and polyketides', 'Biosynthesis of other secondary metabolites', 'Xenobiotics biodegradation and metabolism']

    class_list = []
    for i, path_class in enumerate(p_properties[0]):
        if path_class:
            class_list.append(classes[i])

    pred = {'class': class_list, 'name':name}

    return pred
