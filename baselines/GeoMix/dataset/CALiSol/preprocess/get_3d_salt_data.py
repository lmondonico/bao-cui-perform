import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

smiles_solvents_li1 = {
    'LiPF6': '[P-](F)(F)(F)(F)(F)F', 
    'LiBF4': '[B-](F)(F)(F)F', 
    'LiAsF6': '[As-](F)(F)(F)(F)(F)F', 
    'LiBOB': '[B-]12(OC(=O)C(=O)O1)OC(=O)C(=O)O2', 
    'LiBPFPB': '[B-](OC(C(F)(F)(F))(C(F)(F)(F))CO)(OC(C(F)(F)(F))(C(F)(F)(F))CO)', 
    'LiBMB': '[B-](C(C(C=O)O)C(C=O)O)(C(C(C=O)O)C(C=O)O)', 
}

smiles_solvents_li2 = {
    'LiFSI': '[N-](S(=O)(=O)F)S(=O)(=O)F', 
    'LiTDI': 'C(#N)C1=C(N=C([N-]1)C(F)(F)F)C#N', 
    'LiPDI': 'C(#N)C1=C(N=C([N-]1)C(C(F)(F)F)(F)F)C#N', 
    'LiTFSI': 'C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F', 
    'LiClO4': '[O-]Cl(=O)(=O)=O', 
    'LiCF3SO3': 'C(F)(F)(F)S(=O)(=O)[O-]', 
    'LiN(CF3SO2)2': 'C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F'
}

smiles_solvents_li = {**smiles_solvents_li1, **smiles_solvents_li2}

coord_LiPF6 = np.array([[2.4012630202607497, -1.3863698511426377, 4.15377473376843],
                        [3.2070558124748536, -2.4315010948633624, 3.221704693643671],
                        [3.7092696237187774, -1.2110984447591948, 5.0858447738931885],
                        [1.8990492090168258, -2.6067725012468137, 5.0858447738931885],
                        [2.9034768315046735, -0.1659672010384695, 3.221704693643671],
                        [1.0932564168027243, -1.5616412575260847, 3.221704693643671],
                        [1.5954702280466457, -0.3412386074219128, 5.0858447738931885]
                        ])

coord_LiBF4 = np.array([[2.6323717132202447, -2.5961815404538128e-17, 5.419830965],
                        [1.8181082540127305, -0.33131395164052346, 4.324024753802683],
                        [3.4425735001242734, -1.093112310948022, 5.767638825350924],
                        [1.8181082540127305, 0.3313139516405235, 6.515637176197306],
                        [3.442573500124274, 1.0931123109480223, 5.072023104649065]
                        ])

coord_LiAsF6 = np.array([[2.443145312400522, 1.4105506037838136, 8.64805817243548],
                         [1.6097270989619086, 0.25762083234366556, 7.641943041414048],
                         [3.8583208899663854, 1.2652541446894112, 7.641943041414048],
                         [1.8613879482732711, 2.7087768343183725, 7.641943041414048],
                         [3.2765635258391375, 2.563480375223966, 9.654173303456922],
                         [3.02490267652777, 0.11232437324925884, 9.654173303456922],
                         [1.0279697348346557, 1.5558470628782202, 9.654173303456922]
                         ])

def smiles_to_3d_structure(smiles_dict, output_file):
    """
    Converts a SMILES string to 3D atomic coordinates and atomic numbers.
    Saves the result as a .npz file.

    Args:
        smiles_list (list of str): List of SMILES strings of the molecules.
        output_file (str): Path to save the .npz file.
    """
    name_list = list(smiles_dict.keys())
    smiles_list = list(smiles_dict.values())

    coords_list = []
    atomic_numbers_list = []
    edge_index_list = []
    edge_attr_list = []

    for smiles in smiles_list:
        if smiles == '[P-](F)(F)(F)(F)(F)F':
            coords = coord_LiPF6
            atomic_numbers = [15, 9, 9, 9, 9, 9, 9]
            edge_index = [[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6],
                          [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]]
            edge_attr = [1.0] * 12
        elif smiles == '[B-](F)(F)(F)F':
            coords = coord_LiBF4
            atomic_numbers = [5, 9, 9, 9, 9]
            edge_index = [[0, 1, 0, 2, 0, 3, 0, 4],
                          [1, 0, 2, 0, 3, 0, 4, 0]]
            edge_attr = [1.0] * 10
        elif smiles == '[As-](F)(F)(F)(F)(F)F':
            coords = coord_LiAsF6
            atomic_numbers = [33, 9, 9, 9, 9, 9, 9]
            edge_index = [[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6],
                          [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]]
            edge_attr = [1.0] * 12
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string.")

            mol = AllChem.AddHs(mol)

            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)

            try:
                conf = mol.GetConformer()
            except ValueError as e:
                print("smiles: ", smiles)
                print(e)
                continue
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

            edge_index = [[], []]
            edge_attr = []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index[0].append(start)
                edge_index[0].append(end)
                edge_index[1].append(end)
                edge_index[1].append(start)
                edge_attr.append(bond.GetBondTypeAsDouble())
                edge_attr.append(bond.GetBondTypeAsDouble())

        coords_list.append(coords)
        atomic_numbers_list.append(np.array(atomic_numbers))
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)

    np.savez(output_file, 
             name_list=np.array(name_list, dtype=object), 
             coordinates=np.array(coords_list, dtype=object), 
             atomic_numbers=np.array(atomic_numbers_list, dtype=object), 
             edge_index=np.array(edge_index_list, dtype=object),  
             edge_attr=np.array(edge_attr_list, dtype=object))
    print(f"3D structure with edges saved to {output_file}.")

smiles_input = smiles_solvents_li
output_path = "./data/CoordData/coords-salts-test.npz"
smiles_to_3d_structure(smiles_input, output_path)

data = np.load(output_path, allow_pickle=True)
print("Loaded names:\n", data['name_list'])
print("Loaded 3D coordinates:\n", data['coordinates'])
print("Loaded atomic numbers:\n", data['atomic_numbers'])
print("Loaded edge index:\n", data['edge_index'])
print("edge_index shape", data['edge_index'].shape)
print("Loaded edge attr:\n", data['edge_attr'])
print("edge_attr shape", data['edge_attr'].shape)
