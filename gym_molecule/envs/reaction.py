from rdkit import Chem
from rdkit.Chem import rdChemReactions,AllChem

# def replace_reation_point(warhead):
#     rxn = rdChemReactions.ReactionFromSmarts('[C:1]=[C:2]>>[C:1]=[CH-:2]')
#     reacts = (Chem.MolFromSmiles(warhead))
#     products = rxn.RunReactants(reacts)


def cys_Michael_addition(reactant):
    rxn = rdChemReactions.ReactionFromSmarts('[C:1]=[C:2].[S:3]-[C:4]-[C:5]>>[C:1]-[C:2]-[S:3]-[C:4]-[CH1-2:5]')
    # [C:1]-[C:2]=[C:3].[S:4]-[C:5]-[C:6] >> [C:3]-[C:2]-[C:1]-[S:4]-[C:5]
    reacts = (Chem.MolFromSmiles(reactant),Chem.MolFromSmarts('[S:3]-[C:2]-[C:5]'))
    products = rxn.RunReactants(reacts)
    mol = products[0][0]
    mol_no_att = Chem.DeleteSubstructs(mol,Chem.MolFromSmiles('*'))
    mol_no_att = Chem.MolFromSmiles(Chem.MolToSmiles(mol_no_att))

    patt = Chem.MolFromSmarts('[CH-2]C')
    # mol_no_att.HasSubstructmatch(patt)

    Ca_idx,Cb_idx = mol_no_att.GetSubstructMatch(patt)

    # for atom in mol_no_att.GetAtoms():
    #     if atom.GetSymbol() == 'S':
    #         # S_idx = atom.GetIdx()
    #         for atom_neig in atom.GetNeighbors():
    #             if atom_neig.GetSymbol() == 'C' and len(atom_neig.GetNeighbors())==2:
    #                 C_idx = atom_neig.GetIdx()

    return Chem.MolToSmiles(mol), Chem.MolToSmiles(mol_no_att), [Cb_idx, Ca_idx]



def processing(w_vocab):
    smis_fail, smis, ligindices, ligsmiles, warheads= [], [], [], [], []
    for smi in w_vocab:
        smi = smi.strip('\n')
        try:
            warhead, ligsmile, ligindex = cys_Michael_addition(smi)
            smis.append(smi)
            ligindices.append(ligindex)
            ligsmiles.append(ligsmile)
            warheads.append(warhead)
        except Exception:
            smis_fail.append(smi)

    return warheads, ligsmiles, ligindices, smis_fail


#
# from rdkit.Chem import Draw
# mol = Chem.MolFromSmiles('CSCC(C#N)C=O')
# for atom in mol.GetAtoms():
#     atom.SetProp("atomNote", str(atom.GetIdx()))
# Draw.ShowMol(mol)