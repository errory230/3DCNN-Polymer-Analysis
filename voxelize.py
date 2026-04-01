import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def voxelize_molecule(mol, resolution=0.5, margin=2.0):
    if not mol.GetConformer().Is3D():
        raise ValueError("Molecule does not have 3D coordinates. Run EmbedMolecule and MMFFOptimizeMolecule first.")

    coords = mol.GetConformer().GetPositions()
    min_coords = np.min(coords, axis=0) - margin
    max_coords = np.max(coords, axis=0) + margin

    grid_shape = np.ceil((max_coords - min_coords) / resolution).astype(int)  # 0.5 Angstrom per grid
    voxel_grid = np.zeros(grid_shape, dtype=int)  
    origin = min_coords
    color_grid = np.zeros(grid_shape, dtype=int)

    atom_colors = {'C': 'gray', 'H': 'white', 'O': 'red', 'N': 'blue', 'S': 'yellow', 'P': 'orange', 'F': 'lightgreen',
                   'Cl': 'green', 'Br': 'darkred', 'I': 'purple'}
    color_grid = np.tile([0,0,0,0], color_grid.shape + (1,))

    symbol_encode = ['H', 'C', 'N','S','O']
    atom_grid = np.zeros(grid_shape, dtype=int)
    atom_grid = np.tile([0,0,0,0,0], atom_grid.shape + (1,))

    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtoms()[i]
        atom_coord = coords[i]
        radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
        symbol = atom.GetSymbol()
        color = atom_colors.get(symbol, 'blue')

        min_voxel_idx = np.floor((atom_coord - radius - origin) / resolution).astype(int)
        max_voxel_idx = np.ceil((atom_coord + radius - origin) / resolution).astype(int)

        for x in range(max(0, min_voxel_idx[0]), min(grid_shape[0], max_voxel_idx[0] + 1)):
            for y in range(max(0, min_voxel_idx[1]), min(grid_shape[1], max_voxel_idx[1] + 1)):
                for z in range(max(0, min_voxel_idx[2]), min(grid_shape[2], max_voxel_idx[2] + 1)):
                    voxel_center = origin + (np.array([x, y, z]) + 0.5) * resolution
                    distance = np.linalg.norm(atom_coord - voxel_center)
                    if distance <= radius:
                        voxel_grid[x, y, z] = 1
                        rgba = mcolors.to_rgba(color)
                        rgba_l= [rgba[0], rgba[1], rgba[2], rgba[3]]
                        color_grid[x, y, z] = rgba_l
                        atom_grid[x,y,z, symbol_encode.index(symbol)] = 1
    return voxel_grid, origin, color_grid, atom_grid

def voxelize_smiles(smiles, aug_num):
  mol = Chem.MolFromSmiles(smiles)
  mol = Chem.AddHs(mol)
  cat_vox_data = []
  for u in range(aug_num):
    AllChem.EmbedMolecule(mol, randomSeed = u)
    AllChem.MMFFOptimizeMolecule(mol)
    voxel_grid, origin, color_grid, atom_grid = voxelize_molecule(mol)
    cat_vox_data.append(np.array(atom_grid))
  return cat_vox_data
