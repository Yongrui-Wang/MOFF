import random
import time

from scipy import sparse
import dgl
import gym
from gym_molecule.envs.env_utils_graph import *
from gym_molecule.envs.dock_gpu_non import autodock_gpu
from gym_molecule.envs.rule_of_five import rule_of_five
from gym_molecule.envs.sascorer import calculateScore

from run_rl import molecule_arg_parser

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen

import torch


# 主要行使功能： 分子操作
def adj2sparse(adj):
    """
        adj: [3, 47, 47] float numpy array
        return: a tuple of 2 lists
    """
    adj = [x * (i + 1) for i, x in enumerate(adj)]
    adj = [sparse.dok_matrix(x) for x in adj]

    if not all([adj_i is None for adj_i in adj]):
        adj = sparse.dok_matrix(np.sum(adj))
        adj.setdiag(0)

        all_edges = list(adj.items())
        e_types = np.array([edge[1] - 1 for edge in all_edges], dtype=int)
        e = np.transpose(np.array([list(edge[0]) for edge in all_edges]))

        n_edges = len(all_edges)

        e_x = np.zeros((n_edges, 4))
        e_x[np.arange(n_edges), e_types] = 1
        e_x = torch.Tensor(e_x)
        return e, e_x
    else:
        return None


def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())

    return att_points


def map_idx(idx, idx_list, mol):
    abs_id = idx_list[idx]
    neigh_idx = mol.GetAtomWithIdx(abs_id).GetNeighbors()[0].GetIdx()
    return neigh_idx


def reward_noncovalent(smis, autodock, reward_vina_min=0):
    docking_scores = - np.array(autodock.dock(smis))
    docking_scores = np.clip(docking_scores, reward_vina_min, None)
    lpks = np.array(rule_of_five(smis))

    reward = np.around((docking_scores + lpks), 2)
    # final_reward = docking_scores
    final_reward = [docking_scores, lpks, reward]
    return docking_scores


def reward_covalent(smis, autodock):
    s = time.time()
    docking_scores = -np.array(autodock.dock(smis))
    e = time.time()
    print(e - s)

    return docking_scores

def calculate_qed_sa(smiles_list):
    qed_values = []
    sa_values = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:

            qed = QED.qed(mol)
            qed_values.append(qed)


            sa = calculateScore(mol)
            sa_values.append(sa)
        else:
            qed_values.append(np.nan)
            sa_values.append(np.nan)

    qed_values = np.array(qed_values)
    sa_values = np.array(sa_values)

    return qed_values, sa_values




class MoleculeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def init(self, docking_config=dict(), data_type='zinc', ratios=dict(), reward_step_total=1, is_normalize=0,
             reward_type='crystal', reward_target=0.5, has_scaffold=False, has_feature=False, is_conditional=False,
             conditional='low', max_action=4, min_action=0, force_final=False, is_covalent=False):
        '''
        own init function, since gym does not support passing argument
        '''
        self.is_normalize = bool(is_normalize)
        self.has_feature = has_feature
        self.is_covalent = is_covalent

        # init smi
        self.starting_smi = 'Cl*'
        self.smi = self.starting_smi

        self.mol = Chem.MolFromSmiles(self.smi)
        self.smile_list = []
        self.smile_old_list = []
        # self.warhead_idxs = []

        possible_atoms = ATOM_VOCAB
        possible_motifs = FRAG_VOCAB
        possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        self.atom_type_num = len(possible_atoms)
        self.motif_type_num = len(possible_motifs)
        self.possible_atom_types = np.array(possible_atoms)
        self.possible_motif_types = np.array(possible_motifs)

        self.possible_bond_types = np.array(possible_bonds, dtype=object)

        self.d_n = len(self.possible_atom_types) + 18

        self.max_action = max_action
        self.min_action = min_action

        self.max_atom = 150
        self.action_space = gym.spaces.MultiDiscrete([20, len(FRAG_VOCAB), 20])

        self.counter = 0
        self.level = 0  # for curriculum learning, level starts with 0, and increase afterwards
        self.n = 0  # nth step in each episode

        self.noncovalent_predictor = autodock_gpu(docking_config)
        if is_covalent:
            from gym_molecule.envs.dock_gpu_cov import covdock_gpu
            self.covalent_predictor = covdock_gpu(docking_config)

        self.attach_point = Chem.MolFromSmiles('*')
        self.Na = Chem.MolFromSmiles('[Na+]')
        self.K = Chem.MolFromSmiles('[K+]')
        self.H = Chem.MolFromSmiles('[H+]')
        self.Cl = Chem.MolFromSmiles('Cl')

        # if is_covalent:
        #     self.warhead_idx = -1

    def seed(self, seed):
        np.random.seed(seed=seed)
        random.seed(seed)

    def normalize_adj(self, adj):
        degrees = np.sum(adj, axis=2)

        D = np.zeros((adj.shape[0], adj.shape[1], adj.shape[2]))
        for i in range(D.shape[0]):
            D[i, :, :] = np.diag(np.power(degrees[i, :], -0.5))
        adj_normal = D @ adj @ D
        adj_normal[np.isnan(adj_normal)] = 0
        return adj_normal

    def reset_batch(self):
        self.smile_list = []
        # self.warhead_idxs = []

    def reward_batch(self):
        reward = []
        # print('smiles list', self.smile_list)
        if self.is_covalent:
            return reward_covalent(self.smile_list, self.covalent_predictor)
        else:
            return reward_noncovalent(self.smile_list, self.noncovalent_predictor)

    def reward_single(self, smile_list):
        reward = []
        # print('smiles list', smile_list)
        if self.is_covalent:
            return reward_covalent(self.smile_list, self.covalent_predictor)
        else:
            return reward_noncovalent(self.smile_list, self.noncovalent_predictor)

    def SAandQEDscores(self):
        return calculate_qed_sa(self.smile_list)

    def step(self, ac):
        """
        Perform a given action
        :param action:
        :param action_type:
        :return: reward of 1 if resulting molecule graph does not exceed valency,
        -1 if otherwise
        """
        ac = ac[0]

        ### init
        info = {}
        self.mol_old = copy.deepcopy(self.mol)

        stop = False
        new = False

        if (self.counter >= self.max_action) or get_att_points(self.mol) == []:
            new = True
        else:

            self._add_motif(ac)



        reward_step = 0.05
        if self.mol.GetNumAtoms() > self.mol_old.GetNumAtoms():
            reward_step += 0.05
        self.counter += 1
        self.n += 1

        if new:
            reward = 0
            # Only store for obs if attachment point exists in o2
            if get_att_points(self.mol) != []:
                mol_no_att = self.get_final_mol()

                Chem.SanitizeMol(mol_no_att, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                smi_no_att = Chem.MolToSmiles(mol_no_att)
                info['smile'] = smi_no_att
                # print("smi:", smi_no_att)

                self.smile_list.append(smi_no_att)

                # if self.is_covalent:
                #     self.warhead_idxs.append(self.warhead_idx)

                # Info for old mol
                mol_old_no_att = self.get_final_mol_ob(self.mol_old)
                Chem.SanitizeMol(mol_old_no_att, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                smi_old_no_att = Chem.MolToSmiles(mol_no_att)
                info['old_smi'] = smi_old_no_att
                self.smile_old_list.append(smi_old_no_att)

                stop = True
            else:
                stop = False


        ### use stepwise reward
        else:
            reward = reward_step

        info['stop'] = stop

        # get observation
        ob = self.get_observation()
        return ob, reward, new, info

    def reset(self, smile=None):
        '''
        to avoid error, assume an atom already exists
        :return: ob
        '''
        if smile is not None:
            self.mol = Chem.RWMol(Chem.MolFromSmiles(smile))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            # init smi
            self.smi = self.starting_smi
            self.mol = Chem.MolFromSmiles(self.smi)
            # self.smile_list = [] # only for single motif
        self.counter = 0
        self.n = 0
        ob = self.get_observation()
        return ob

    def render(self, mode='human', close=False):
        return

    def sample_motif(self, n, step_list):

        go_on = True
        while go_on:
            cur_mol_atts = get_att_points(self.mol)

            ac1 = np.random.randint(len(cur_mol_atts))
            if n >= len(step_list):
                ac2 = np.random.randint(self.motif_type_num)
            else:
                ac2 = np.random.randint(VOCAB_NUM[step_list[n]]) + sum(VOCAB_NUM[:step_list[n]])
            motif = FRAG_VOCAB_MOL[ac2]
            ac3 = np.random.randint(len(FRAG_VOCAB_ATT[ac2]))
            a = self.action_space.sample()

            a[0] = ac1
            a[1] = ac2
            a[2] = ac3

            go_on = False

        return a

    def _add_motif(self, ac):

        cur_mol = Chem.ReplaceSubstructs(self.mol, self.attach_point, self.Na)[ac[0]]
        motif = FRAG_VOCAB_MOL[ac[1]]
        att_point = FRAG_VOCAB_ATT[ac[1]]
        # motif_atom = map_idx(ac[2], att_point, motif)
        motif = Chem.ReplaceSubstructs(motif, self.attach_point, self.K)[ac[2]]
        # if self.n == 1: # delete other attach point in hg
        #     motif = Chem.DeleteSubstructs(motif, Chem.MolFromSmiles("*"))

        # if self.n == 2:
        #     cur_mol = Chem.DeleteSubstructs(cur_mol, Chem.MolFromSmiles("*"))

        for atom in motif.GetAtoms():
            if atom.GetSymbol() == 'K':
                motif_atom = map_idx(0, [atom.GetIdx()], motif)

        motif = Chem.DeleteSubstructs(motif, self.K)
        next_mol = Chem.ReplaceSubstructs(cur_mol, self.Na, motif, replacementConnectionPoint=motif_atom)[0]
        if self.n == 0:
            next_mol = Chem.ReplaceSubstructs(next_mol, self.Cl, Chem.MolFromSmiles("*"))[0]
        # if self.n == 3:
        #     next_mol_w = Chem.RWMol(next_mol)
        #     next_mol_w.AddAtom(self.attach_point.GetAtoms()[0])
        #     next_mol = next_mol_w.GetMol()

        # To address the state of a completed molecule without available attachment points
        if self.n == (self.max_action - 1) and get_att_points(next_mol) == []:
            next_mol = Chem.CombineMols(next_mol, self.attach_point)

        self.mol = next_mol

    def get_final_smiles_mol(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
        m = convert_radical_electrons_to_hydrogens(m)
        return m, Chem.MolToSmiles(m, isomericSmiles=True)

    def get_final_mol(self):
        """
        Returns a rdkit mol object of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
        return m

    def get_final_mol_ob(self, mol):
        m = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
        return m

    def get_observation(self, expert_smi=None):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        ob = {}

        if expert_smi:
            mol = Chem.MolFromSmiles(expert_smi)
        else:
            ob['att'] = get_att_points(self.mol)
            mol = copy.deepcopy(self.mol)

        try:
            Chem.SanitizeMol(mol)
        except:
            pass

        smi = Chem.MolToSmiles(mol)

        n = mol.GetNumAtoms()
        F = np.zeros((1, self.max_atom, self.d_n))

        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()

            atom_symbol = a.GetSymbol()
            if self.has_feature:
                float_array = atom_feature(a, use_atom_meta=True)
            else:
                float_array = (atom_symbol == self.possible_atom_types).astype(float)

            F[0, atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))

        for b in mol.GetBonds():
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)
            try:
                assert float_array.sum() != 0
            except:
                print('error', bond_type)
            E[:, begin_idx, end_idx] = float_array

        if self.is_normalize:
            E = self.normalize_adj(E)

        ob_adj = adj2sparse(E.squeeze())
        ob_node = torch.Tensor(F)
        g = dgl.DGLGraph()

        ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
        g.add_nodes(ob_len)
        if ob_adj is not None and len(ob_adj[0]) > 0:
            g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
        g.ndata['x'] = ob_node[:, :int(ob_len), :].squeeze(0)

        if molecule_arg_parser().parse_args().gnn_type == "GAT":
            g = dgl.add_self_loop(g)

        ob['g'] = g
        ob['smi'] = smi

        return ob

    def get_observation_mol(self, mol):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        ob = {}

        ob['att'] = get_att_points(mol)

        try:
            Chem.SanitizeMol(mol)
        except:
            pass

        smi = Chem.MolToSmiles(mol)

        n = mol.GetNumAtoms()
        F = np.zeros((1, self.max_atom, self.d_n))

        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()

            atom_symbol = a.GetSymbol()
            if self.has_feature:
                float_array = atom_feature(a, use_atom_meta=True)
            else:
                float_array = (atom_symbol == self.possible_atom_types).astype(float)

            F[0, atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))

        for b in mol.GetBonds():

            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)

            try:
                assert float_array.sum() != 0
            except:
                print('error', bond_type)
            E[:, begin_idx, end_idx] = float_array

        if self.is_normalize:
            E = self.normalize_adj(E)

        ob_adj = adj2sparse(E.squeeze())
        ob_node = torch.Tensor(F)
        g = dgl.DGLGraph()

        ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
        g.add_nodes(ob_len)
        if ob_adj is not None and len(ob_adj[0]) > 0:
            g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
        g.ndata['x'] = ob_node[:, :int(ob_len), :].squeeze(0)

        if molecule_arg_parser().parse_args().gnn_type == "GAT":
            g = dgl.add_self_loop(g)

        ob['g'] = g
        ob['smi'] = smi
        return ob