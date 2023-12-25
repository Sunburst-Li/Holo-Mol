from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class Dataset():
    def __init__(self, dataset, batch=128, fold=1):
        self.dataset = dataset
        self.fold = fold
        self.task = "regression"
        self.target_name = ['kOH_rscale','Group']
        self.max_atoms = 0
        self.batch = batch
        self.mols = []
        self.coords = []
        self.target = []
        self.smiles = []
        self.x, self.c, self.y, self.s = {}, {}, {}, {}
        self.use_atom_symbol = True
        self.use_degree = True
        self.use_hybridization = True
        self.use_implicit_valence = True
        self.use_partial_charge = False
        self.use_formal_charge = True
        self.use_ring_size = True
        self.use_hydrogen_bonding = True
        self.use_acid_base = True
        self.use_aromaticity = True
        self.use_chirality = True
        self.use_num_hydrogen = True
        self.load_dataset()
        mp = MPGenerator([], [], [],[], 1,
                         use_atom_symbol=self.use_atom_symbol,
                         use_degree=self.use_degree,
                         use_hybridization=self.use_hybridization,
                         use_implicit_valence=self.use_implicit_valence,
                         use_partial_charge=self.use_partial_charge,
                         use_formal_charge=self.use_formal_charge,
                         use_ring_size=self.use_ring_size,
                         use_hydrogen_bonding=self.use_hydrogen_bonding,
                         use_acid_base=self.use_acid_base,
                         use_aromaticity=self.use_aromaticity,
                         use_chirality=self.use_chirality,
                         use_num_hydrogen=self.use_num_hydrogen)
        self.num_features = mp.get_num_features()
        self.mean = 0
        self.std = 1
     

    def load_dataset(self):
        if self.dataset == "group":
            self.task = "regression"
            self.target_name = ['kOH_rscale','Group']
        else:
            pass
        mols_train = []
        coords_train = []
        target_train = []
        smiles_train= []
        mols_test = []
        coords_test = []
        target_test = []
        smiles_test = []
        
        for groupnum in range(1, self.fold + 1):
            self.path = "../data/{}{}.sdf".format(self.dataset, groupnum)
            print()
            print("="*100)
            print(self.path)
            print(os.getcwd())
            mols = Chem.SDMolSupplier(self.path)
            x, c, y, s = [], [], [], []
            x_test, c_test, y_test, s_test = [], [], [], []
            x_train, c_train, y_train, s_train = [], [], [], []
            for mol in mols:
                if mol is not None:
                    if type(self.target_name) is list:
                        if mol.GetProp(self.target_name[1] + str(groupnum)) == 'train':
                            y_train.append(float(mol.GetProp(self.target_name[0])))
                            s_train.append(Chem.MolToSmiles(mol))
                            x_train.append(mol)
                            c_train.append(mol.GetConformer().GetPositions())
                        elif mol.GetProp(self.target_name[1] + str(groupnum)) == 'test':
                            y_test.append(float(mol.GetProp(self.target_name[0])))
                            s_test.append(Chem.MolToSmiles(mol))
                            x_test.append(mol)
                            c_test.append(mol.GetConformer().GetPositions())
                    else:
                        continue
            x = x + x_train + x_test
            c = c + c_train + c_test
            y = y + y_train + y_test
            s = s + s_train + s_test
            train_len = len(x_train)
            test_len = len(x_test)
            assert len(x) == len(y)
            new_x, new_c, new_y = [], [], []
            if self.max_atoms > 0:
                for mol, coo, tar in zip(x, c, y):
                    if mol.GetNumAtoms() <= self.max_atoms:
                        new_x.append(mol)
                        new_c.append(coo)
                        new_y.append(tar)
                x = new_x
                c = new_c
                y = new_y
            else:
                for mol, tar in zip(x, y):
                    self.max_atoms = max(self.max_atoms, mol.GetNumAtoms())
            self.mols, self.coords, self.target, self.smiles  = np.array(x), np.array(c), np.array(y),np.array(s)
            train_mols = self.mols[:train_len]
            train_coords = self.coords[:train_len]
            train_target = self.target[:train_len]
            train_smiles = self.smiles[:train_len]
            test_mols = self.mols[train_len:test_len + train_len]
            test_coords = self.coords[train_len:test_len + train_len]
            test_target = self.target[train_len:test_len + train_len]
            test_smiles = self.smiles[train_len:test_len + train_len]
            mols_train.append(train_mols)
            coords_train.append(train_coords)
            target_train.append(train_target)
            smiles_train.append(train_smiles)
            mols_test.append(test_mols)
            coords_test.append(test_coords)
            target_test.append(test_target)
            smiles_test.append(test_smiles)
        self.x_original = {"train": mols_train,
                  "test": mols_test}
        self.c_original = {"train": coords_train,
                  "test": coords_test}
        self.y_original = {"train": target_train,
                  "test": target_test}
        self.s_original = {"train": smiles_train,
                  "test": smiles_test}
        self.x = {"train": None,
                  "test": None}
        self.c = {"train": None,
                  "test": None}
        self.y = {"train": None,
                  "test": None}
        self.s = {"train": None,
                  "test": None}
        
    def save_dataset(self, path, pred=None, target="test", filename=None):
        mols = []
        for idx, (x, c, y) in enumerate(zip(self.x[target], self.c[target], self.y[target])):
            x.SetProp("true", str(y * self.std + self.mean))
            if pred is not None:
                x.SetProp("pred", str(pred[idx][0] * self.std + self.mean))
            mols.append(x)
        if filename is not None:
            w = Chem.SDWriter(path + filename + ".sdf")
        else:
            w = Chem.SDWriter(path + target + ".sdf")
        for mol in mols:
            if mol is not None:
                w.write(mol)

    def replace_dataset(self, path, subset="test", target_name="target"):
        x, c, y = [], [], []
        mols = Chem.SDMolSupplier(path)
        for mol in mols:
            if mol is not None:
                if type(target_name) is list:
                    y.append([float(mol.GetProp(t)) if t in mol.GetPropNames() else -1 for t in target_name])
                elif target_name in mol.GetPropNames():
                    _y = float(mol.GetProp(target_name))
                    if _y == -1:
                        continue
                    else:
                        y.append(_y)
                else:
                    continue
                x.append(mol)
                c.append(mol.GetConformer().GetPositions())
        x = np.array(x)
        c = np.array(c)
        y = (np.array(y) - self.mean) / self.std
        self.x[subset] = x
        self.c[subset] = c
        self.y[subset] = y.astype(int) if self.task != "regression" else y

    def set_features(self, use_atom_symbol=True, use_degree=True, use_hybridization=True, use_implicit_valence=True,
                     use_partial_charge=False, use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True,
                     use_acid_base=True, use_aromaticity=True, use_chirality=True, use_num_hydrogen=True):
        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen
        mp = MPGenerator([], [], [],[], 1,
                         use_atom_symbol=self.use_atom_symbol,
                         use_degree=self.use_degree,
                         use_hybridization=self.use_hybridization,
                         use_implicit_valence=self.use_implicit_valence,
                         use_partial_charge=self.use_partial_charge,
                         use_formal_charge=self.use_formal_charge,
                         use_ring_size=self.use_ring_size,
                         use_hydrogen_bonding=self.use_hydrogen_bonding,
                         use_acid_base=self.use_acid_base,
                         use_aromaticity=self.use_aromaticity,
                         use_chirality=self.use_chirality,
                         use_num_hydrogen=self.use_num_hydrogen)
        self.num_features = mp.get_num_features()

    def generator(self, target, task=None):
        return MPGenerator(self.x[target], self.c[target], self.y[target],self.s[target], self.batch,
                           task=task if task is not None else self.task,
                           num_atoms=self.max_atoms,
                           use_atom_symbol=self.use_atom_symbol,
                           use_degree=self.use_degree,
                           use_hybridization=self.use_hybridization,
                           use_implicit_valence=self.use_implicit_valence,
                           use_partial_charge=self.use_partial_charge,
                           use_formal_charge=self.use_formal_charge,
                           use_ring_size=self.use_ring_size,
                           use_hydrogen_bonding=self.use_hydrogen_bonding,
                           use_acid_base=self.use_acid_base,
                           use_aromaticity=self.use_aromaticity,
                           use_chirality=self.use_chirality,
                           use_num_hydrogen=self.use_num_hydrogen)


class MPGenerator(data.Dataset):
    def __init__(self, x_set, c_set, y_set,s_set, batch, task="binary", num_atoms=0,
                 use_degree=True, use_hybridization=True, use_implicit_valence=True, use_partial_charge=False,
                 use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True, use_acid_base=True,
                 use_aromaticity=True, use_chirality=True, use_num_hydrogen=True, use_atom_symbol=True):
        self.x, self.c, self.y,self.s = x_set, c_set, y_set,s_set
        self.batch = batch
        self.task = task
        self.num_atoms = num_atoms
        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen
        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch:(idx + 1) * self.batch]
        batch_c = self.c[idx * self.batch:(idx + 1) * self.batch]
        batch_y = self.y[idx * self.batch:(idx + 1) * self.batch]
        batch_s = range(idx * self.batch,(idx + 1) * self.batch)
        datas = self.tensorize(batch_x, batch_c)
        return (datas, torch.as_tensor(np.array(batch_y)).float(),torch.as_tensor(np.array(batch_s)))


    def tensorize(self, batch_x, batch_c):
        atom_tensor = np.zeros((len(batch_x), self.num_atoms, self.get_num_features()))
        adjm_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms))
        posn_tensor = np.zeros((len(batch_x), self.num_atoms, self.num_atoms, 3))
        for mol_idx, mol in enumerate(batch_x):
            Chem.RemoveHs(mol)
            mol_atoms = mol.GetNumAtoms()
            atom_tensor[mol_idx, :mol_atoms, :] = self.get_atom_features(mol)
            adjms = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype="float")
            adjms += np.eye(mol_atoms)
            degree = np.array(adjms.sum(1))
            deg_inv_sqrt = np.power(degree, -0.5)
            deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(deg_inv_sqrt)
            adjms = np.matmul(np.matmul(deg_inv_sqrt, adjms), deg_inv_sqrt)
            adjm_tensor[mol_idx, : mol_atoms, : mol_atoms] = adjms
            for atom_idx in range(mol_atoms):
                pos_c = batch_c[mol_idx][atom_idx]
                for neighbor_idx in range(mol_atoms):
                    pos_n = batch_c[mol_idx][neighbor_idx]
                    n_to_c = [pos_c[0] - pos_n[0], pos_c[1] - pos_n[1], pos_c[2] - pos_n[2]]
                    posn_tensor[mol_idx, atom_idx, neighbor_idx, :] = n_to_c
        return torch.as_tensor(atom_tensor).float(), torch.as_tensor(adjm_tensor).float(), torch.as_tensor(posn_tensor).float()

    def get_num_features(self):
        mol = Chem.MolFromSmiles("CC")
        return len(self.get_atom_features(mol)[0])

    def get_atom_features(self, mol):
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)
        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())
        ring = mol.GetRingInfo()
        m = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            o = []
            o += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                            'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']) if self.use_atom_symbol else []
            o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) if self.use_degree else []
            o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                   Chem.rdchem.HybridizationType.SP2,
                                                   Chem.rdchem.HybridizationType.SP3,
                                                   Chem.rdchem.HybridizationType.SP3D,
                                                   Chem.rdchem.HybridizationType.SP3D2]) if self.use_hybridization else []
            o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
            o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_degree else []
            o += [atom.GetIsAromatic()] if self.use_aromaticity else []
            o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                  ring.IsAtomInRingOfSize(atom_idx, 4),
                  ring.IsAtomInRingOfSize(atom_idx, 5),
                  ring.IsAtomInRingOfSize(atom_idx, 6),
                  ring.IsAtomInRingOfSize(atom_idx, 7),
                  ring.IsAtomInRingOfSize(atom_idx, 8)] if self.use_ring_size else []
            o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []
            if self.use_chirality:
                try:
                    o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
                except:
                    o += [False, False] + [atom.HasProp("_ChiralityPossible")]
            if self.use_hydrogen_bonding:
                o += [atom_idx in hydrogen_donor_match]
                o += [atom_idx in hydrogen_acceptor_match]
            if self.use_acid_base:
                o += [atom_idx in acidic_match]
                o += [atom_idx in basic_match]
            m.append(o)
        return np.array(m, dtype=float)
