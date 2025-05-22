from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling



from rdkit import Chem

from gym_molecule.envs.env_utils_graph import ATOM_VOCAB, FRAG_VOCAB, FRAG_VOCAB_MOL, VOCAB_NUM

from .descriptors import ecfp, rdkit_descriptors


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# DGL operations
msg = fn.copy_src(src='x', out='m')


def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'x': accum}


def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'x': accum}


class GNNPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = GNNEmbed(args)
        self.pred_layer = nn.Sequential(
                    nn.Linear(args.emb_size*2, args.emb_size, bias=False),
                    nn.ReLU(inplace=False),
                    nn.Linear(args.emb_size, 1, bias=True))

    def forward(self, o):
        _, _, graph_emb = self.embed(o)
        pred = self.pred_layer(graph_emb)
        return pred


class GNNActorCritic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        # build policy and value functions
        self.embed = GNNEmbed(args)
        self.env = env
        self.pi = SFSPolicy(env, args)
        self.q1 = GNNQFunction(args)
        self.q2 = GNNQFunction(args, override_seed=True)
        self.p = GNNPredictor(args)
        self.cand = self.create_candidate_motifs()

    def create_candidate_motifs(self):
        motif_gs = [self.env.get_observation_mol(mol) for mol in FRAG_VOCAB_MOL]
        return motif_gs

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            o_g, o_n_emb, o_g_emb = self.embed(obs)
            cands = self.embed(deepcopy(self.cand))
            a, _, _ = self.pi(o_g_emb, o_n_emb, o_g, cands, deterministic)
        return a

class GNNQFunction(nn.Module):
    def __init__(self, args, override_seed=False):
        super().__init__()
        if override_seed:
            seed = args.seed + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.batch_size = args.batch_size
        self.device = args.device
        self.emb_size = args.emb_size
        self.max_action2 = len(ATOM_VOCAB)
        self.max_action_stop = 2

        self.d = 2 * args.emb_size + len(FRAG_VOCAB) + 80 
        self.out_dim = 1
        
        self.qpred_layer = nn.Sequential(
                            nn.Linear(self.d, int(self.d//2), bias=False),
                            nn.ReLU(inplace=False),
                            nn.Linear(int(self.d//2), self.out_dim, bias=True))
    
    def forward(self, graph_emb, ac_first_prob, ac_second_hot, ac_third_prob):
        emb_state_action = torch.cat([graph_emb, ac_first_prob, ac_second_hot, ac_third_prob], dim=-1).contiguous()
        qpred = self.qpred_layer(emb_state_action)
        return qpred

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

class SFSPolicy(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.ac_dim = len(FRAG_VOCAB)-1
        self.emb_size = args.emb_size
        self.tau = args.tau
        self.step_list = args.step_list



        self.env = env # env utilized to init cand motif mols
        self.cand = self.create_candidate_motifs()
        self.cand_g = dgl.batch([x['g'] for x in self.cand])
        self.cand_ob_len = self.cand_g.batch_num_nodes().tolist()

        # Create candidate descriptors

        if args.desc == 'ecfp':
            desc = ecfp
            self.desc_dim = 1024
        elif args.desc == 'desc':
            desc = rdkit_descriptors
            self.desc_dim = 199
        self.cand_desc = torch.Tensor([desc(Chem.MolFromSmiles(x['smi'])) # 为什么不使用gcn？
                                for x in self.cand]).to(self.device)
        self.motif_type_num = len(self.cand)

        self.action1_layers = nn.ModuleList([nn.Bilinear(2*args.emb_size, 2*args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size//2, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size//2, 1, bias=True)).to(self.device)])

        # 或许修改为4个更有说服力？
        self.action2_layers = nn.ModuleList([nn.Bilinear(self.desc_dim,args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(self.desc_dim, args.emb_size, bias=False).to(self.device),
                                nn.Linear(args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, args.emb_size, bias=True),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, 1, bias=True),
                                )])

        self.action3_layers = nn.ModuleList([nn.Bilinear(2*args.emb_size, 2*args.emb_size, args.emb_size).to(self.device),
                               nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                               nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                               nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size//2, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size//2, 1, bias=True)).to(self.device)])

        # Zero padding with max number of actions
        self.max_action = 40 # max atoms
        
        print('number of candidate motifs : ', len(self.cand))
        self.ac3_att_len = torch.LongTensor([len(x['att']) 
                                for x in self.cand]).to(self.device)
        self.ac3_att_mask = torch.cat([torch.LongTensor([i]*len(x['att'])) 
                                for i, x in enumerate(self.cand)], dim=0).to(self.device)

    def create_candidate_motifs(self):
        motif_gs = [self.env.get_observation_mol(mol) for mol in FRAG_VOCAB_MOL]
        return motif_gs


    def gumbel_softmax(self, logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, \
                    g_ratio: float = 1e-3) -> torch.Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels * g_ratio) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
        
        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, graph_emb, node_emb, g, cands, n, deterministic=False):
        """
        graph_emb : bs x hidden_dim
        node_emb : (bs x num_nodes) x hidden_dim)
        g: batched graph
        att: indexs of attachment points, list of list
        """
        
        g.ndata['node_emb'] = node_emb
        cand_g, cand_node_emb, cand_graph_emb = cands 

        # Only acquire node embeddings with attatchment points
        ob_len = g.batch_num_nodes().tolist()
        att_mask = g.ndata['att_mask'] # used to select att embs from node embs
        
        if g.batch_size != 1:
            att_mask_split = torch.split(att_mask, ob_len, dim=0)
            att_len = [torch.sum(x, dim=0) for x in att_mask_split]
        else:
            att_len = torch.sum(att_mask, dim=-1) # used to torch.split for att embs

        cand_att_mask = cand_g.ndata['att_mask']
        cand_att_mask_split = torch.split(cand_att_mask, self.cand_ob_len, dim=0)
        cand_att_len = [torch.sum(x, dim=0) for x in cand_att_mask_split]

        # =============================== 
        # step 1 : where to add
        # =============================== 
        # select only nodes with attachment points
        att_emb = torch.masked_select(node_emb , att_mask.unsqueeze(-1))
        att_emb = att_emb.view(-1, 2*self.emb_size)
        
        if g.batch_size != 1:
            graph_expand = torch.cat([graph_emb[i].unsqueeze(0).repeat(att_len[i],1) for i in range(g.batch_size)], dim=0).contiguous()
        else:
            graph_expand = graph_emb.repeat(att_len, 1)

        att_emb = self.action1_layers[0](att_emb, graph_expand) + self.action1_layers[1](att_emb) \
                    + self.action1_layers[2](graph_expand)
        logits_first = self.action1_layers[3](att_emb)

        if g.batch_size != 1:
            ac_first_prob = [torch.softmax(logit, dim=0)
                            for i, logit in enumerate(torch.split(logits_first, att_len, dim=0))]
            ac_first_prob = [p+1e-8 for p in ac_first_prob]
            log_ac_first_prob = [x.log() for x in ac_first_prob]

        else:
            ac_first_prob = torch.softmax(logits_first, dim=0) + 1e-8
            log_ac_first_prob = ac_first_prob.log()

        if g.batch_size != 1:  
            first_stack = []
            first_ac_stack = []
            for i, node_emb_i in enumerate(torch.split(att_emb, att_len, dim=0)):
                ac_first_hot_i = self.gumbel_softmax(ac_first_prob[i], tau=self.tau, hard=True, dim=0).transpose(0,1)
                ac_first_i = torch.argmax(ac_first_hot_i, dim=-1)
                first_stack.append(torch.matmul(ac_first_hot_i, node_emb_i))
                first_ac_stack.append(ac_first_i)

            emb_first = torch.stack(first_stack, dim=0).squeeze(1)
            ac_first = torch.stack(first_ac_stack, dim=0).squeeze(1)
            

            ac_first_prob = torch.cat([
                                torch.cat([ac_first_prob_i, ac_first_prob_i.new_zeros(
                                    max(self.max_action - ac_first_prob_i.size(0),0),1)]
                                        , 0).contiguous().view(1,self.max_action)
                                for i, ac_first_prob_i in enumerate(ac_first_prob)], dim=0).contiguous()

            log_ac_first_prob = torch.cat([
                                    torch.cat([log_ac_first_prob_i, log_ac_first_prob_i.new_zeros(
                                        max(self.max_action - log_ac_first_prob_i.size(0),0),1)]
                                            , 0).contiguous().view(1,self.max_action)
                                    for i, log_ac_first_prob_i in enumerate(log_ac_first_prob)], dim=0).contiguous()
            
        else:            
            ac_first_hot = self.gumbel_softmax(ac_first_prob, tau=self.tau, hard=True, dim=0).transpose(0,1)
            ac_first = torch.argmax(ac_first_hot, dim=-1)
            emb_first = torch.matmul(ac_first_hot, att_emb)
            ac_first_prob = torch.cat([ac_first_prob, ac_first_prob.new_zeros(
                            max(self.max_action - ac_first_prob.size(0),0),1)]
                                , 0).contiguous().view(1,self.max_action)
            log_ac_first_prob = torch.cat([log_ac_first_prob, log_ac_first_prob.new_zeros(
                            max(self.max_action - log_ac_first_prob.size(0),0),1)]
                                , 0).contiguous().view(1,self.max_action)

        # =============================== 
        # step 2 : which motif to add - Using Descriptors
        # ===============================   
        emb_first_expand = emb_first.view(-1, 1, self.emb_size).repeat(1, self.motif_type_num, 1)
        cand_expand = self.cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) + \
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)

        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)

        logit_mask_second = self.mask(logit_second, n)  # mask invaild actions

        ac_second_prob = F.softmax(logit_mask_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_second = torch.matmul(ac_second_hot, cand_graph_emb)
        ac_second = torch.argmax(ac_second_hot, dim=-1)

        # Print gumbel otuput
        ac_second_gumbel = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=False, g_ratio=1e-3)                                    
        
        # ===============================  
        # step 4 : where to add on motif
        # ===============================
        # Select att points from candidate
        cand_att_emb = torch.masked_select(cand_node_emb, cand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1) # bs x (num cands * num att size)
        ac3_att_mask = torch.where(ac3_att_mask==ac_second.view(-1,1),
                            1, 0).view(g.batch_size, -1) # (num_cands * num_nodes)
        ac3_att_mask = ac3_att_mask.bool()

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)#.view(1, -1, 2*self.emb_size)
        
        ac3_att_len = torch.index_select(self.ac3_att_len, 0, ac_second).tolist()
        emb_second_expand = torch.cat([emb_second[i].unsqueeze(0).repeat(ac3_att_len[i],1) for i in range(g.batch_size)]).contiguous()

        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand) \
                  + self.action3_layers[2](ac3_cand_emb)
        
        logits_third = self.action3_layers[3](emb_cat_ac3)

        # predict logit
        if g.batch_size != 1:
            ac_third_prob = [torch.softmax(logit,dim=-1)
                            for i, logit in enumerate(torch.split(logits_third.squeeze(-1), ac3_att_len, dim=0))]
            ac_third_prob = [p+1e-8 for p in ac_third_prob]
            log_ac_third_prob = [x.log() for x in ac_third_prob]

        else:
            logits_third = logits_third.transpose(1,0)
            ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
            log_ac_third_prob = ac_third_prob.log()
        
        # gumbel softmax sampling and zero-padding
        if g.batch_size != 1:
            third_stack = []
            third_ac_stack = []
            for i, node_emb_i in enumerate(torch.split(emb_cat_ac3, ac3_att_len, dim=0)):
                ac_third_hot_i = self.gumbel_softmax(ac_third_prob[i], tau=self.tau, hard=True, dim=-1)
                ac_third_i = torch.argmax(ac_third_hot_i, dim=-1)
                third_stack.append(torch.matmul(ac_third_hot_i, node_emb_i))
                third_ac_stack.append(ac_third_i)

                del ac_third_hot_i
            emb_third = torch.stack(third_stack, dim=0).squeeze(1)
            ac_third = torch.stack(third_ac_stack, dim=0)
            ac_third_prob = torch.cat([
                                torch.cat([ac_third_prob_i, ac_third_prob_i.new_zeros(
                                    self.max_action - ac_third_prob_i.size(0))]
                                        , dim=0).contiguous().view(1,self.max_action)
                                for i, ac_third_prob_i in enumerate(ac_third_prob)], dim=0).contiguous()
            
            log_ac_third_prob = torch.cat([
                                    torch.cat([log_ac_third_prob_i, log_ac_third_prob_i.new_zeros(
                                        self.max_action - log_ac_third_prob_i.size(0))]
                                            , 0).contiguous().view(1,self.max_action)
                                    for i, log_ac_third_prob_i in enumerate(log_ac_third_prob)], dim=0).contiguous()

        else:
            ac_third_hot = self.gumbel_softmax(ac_third_prob, tau=self.tau, hard=True, dim=-1)
            ac_third = torch.argmax(ac_third_hot, dim=-1)
            emb_third = torch.matmul(ac_third_hot, emb_cat_ac3)
            
            ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
            log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

        # ==== concat everything ====

        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob, 
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()
        ac = torch.stack([ac_first, ac_second, ac_third], dim=1)

        return ac, (ac_prob, log_ac_prob), (ac_first_prob, ac_second_hot, ac_third_prob)
    
    def sample(self, ac, graph_emb, node_emb, g, cands, n):
        g.ndata['node_emb'] = node_emb
        cand_g, cand_node_emb, cand_graph_emb = cands 

        # Only acquire node embeddings with attatchment points
        ob_len = g.batch_num_nodes().tolist()
        att_mask = g.ndata['att_mask'] # used to select att embs from node embs
        att_len = torch.sum(att_mask, dim=-1) # used to torch.split for att embs

        cand_att_mask = cand_g.ndata['att_mask']
        cand_att_mask_split = torch.split(cand_att_mask, self.cand_ob_len, dim=0)
        cand_att_len = [torch.sum(x, dim=0) for x in cand_att_mask_split]

        # =============================== 
        # step 1 : where to add
        # =============================== 
        # select only nodes with attachment points
        att_emb = torch.masked_select(node_emb, att_mask.unsqueeze(-1))
        att_emb = att_emb.view(-1, 2*self.emb_size)
        graph_expand = graph_emb.repeat(att_len, 1)
        
        att_emb = self.action1_layers[0](att_emb, graph_expand) + self.action1_layers[1](att_emb) \
                    + self.action1_layers[2](graph_expand)
        logits_first = self.action1_layers[3](att_emb).transpose(1,0)
            
        ac_first_prob = torch.softmax(logits_first, dim=-1) + 1e-8
        
        log_ac_first_prob = ac_first_prob.log()
        ac_first_prob = torch.cat([ac_first_prob, ac_first_prob.new_zeros(1,
                        max(self.max_action - ac_first_prob.size(1),0))]
                            , 1).contiguous()
        
        log_ac_first_prob = torch.cat([log_ac_first_prob, log_ac_first_prob.new_zeros(1,
                        max(self.max_action - log_ac_first_prob.size(1),0))]
                            , 1).contiguous()
        emb_first = att_emb[ac[0]].unsqueeze(0)
        
        # =============================== 
        # step 2 : which motif to add     
        # ===============================   
        emb_first_expand = emb_first.repeat(1, self.motif_type_num, 1)
        cand_expand = self.cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)     
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) + \
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)

        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)
        logit_mask_second = self.mask(logit_second, n) # maslogit_second = {Tensor}  Show Valuek invaild actions

        ac_second_prob = F.softmax(logit_mask_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_second = torch.matmul(ac_second_hot, cand_graph_emb)
        ac_second = torch.argmax(ac_second_hot, dim=-1)

        # ===============================  
        # step 3 : where to add on motif
        # ===============================
        # Select att points from candidates
        
        cand_att_emb = torch.masked_select(cand_node_emb, cand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1) # bs x (num cands * num att size)
        # torch where currently does not support cpu ops    
        
        ac3_att_mask = torch.where(ac3_att_mask==ac[1], 
                            1, 0).view(g.batch_size, -1) # (num_cands * num_nodes)
        ac3_att_mask = ac3_att_mask.bool()

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)
        
        ac3_att_len = self.ac3_att_len[ac[1]]
        emb_second_expand = emb_second.repeat(ac3_att_len,1)
        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand) \
                  + self.action3_layers[2](ac3_cand_emb)
        logits_third = self.action3_layers[3](emb_cat_ac3)
        logits_third = logits_third.transpose(1,0)
        ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
        log_ac_third_prob = ac_third_prob.log()

        # gumbel softmax sampling and zero-padding
        emb_third = emb_cat_ac3[ac[2]].unsqueeze(0)
        ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
        log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

        # ==== concat everything ====
        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob, 
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()

        return (ac_prob, log_ac_prob), (ac_first_prob, ac_second_hot, ac_third_prob)

    def mask(self, logits, t):
        logits_masks = []
        if isinstance(t, list):
            for i in range(len(t)):
                mask_l = self.get_mask(t[i]).to(self.device)
                logits_masks.append(torch.where(mask_l, logits[i], torch.tensor(-1e+8).to(self.device)))
            logits_masks = torch.stack(logits_masks, dim=0)
        else:
            mask_l = self.get_mask(t).to(self.device)
            logits_masks.append(torch.where(mask_l, logits, torch.tensor(-1e+8).to(self.device)))
            logits_masks = torch.cat(logits_masks, dim=0)

        return logits_masks
    
    def get_mask(self,t):  # time step
        num_fraglib = len(VOCAB_NUM)
        mask_list = [torch.zeros(VOCAB_NUM[i]) for i in range(num_fraglib)]
        if t >= len(self.step_list):
            mask_list = torch.ones(sum(VOCAB_NUM)).type(torch.BoolTensor)
        else:

            mask_list[self.step_list[t]] = torch.ones(VOCAB_NUM[self.step_list[t]])
            mask_list = torch.cat(mask_list).type(torch.BoolTensor)
        return mask_list


class GNNEmbed(nn.Module):
    def __init__(self, args):

        ### GCN
        super().__init__()
        self.device = args.device
        self.d_n = len(ATOM_VOCAB)+18
        
        self.emb_size = args.emb_size * 2
        self.gnn_aggregate = args.gnn_aggregate

        in_channels = 8
        self.emb_linear = nn.Linear(self.d_n, in_channels, bias=False)

        self.gnn_type = args.gnn_type
        assert args.gnn_type in ['GCN', 'GIN', 'GAT'], "Wrong gcn type"
        assert args.gnn_aggregate in ['sum', 'gmt'], "Wrong gcn agg type"
        if self.gnn_type == "GCN":
            from models.models_dgl import GCNLayer
            self.gnn_layers = nn.ModuleList([GCNLayer(in_channels, self.emb_size, agg="sum", residual=False)])
            for _ in range(args.layer_num_g-1):
                self.gnn_layers.append(GCNLayer(self.emb_size, self.emb_size, agg="sum"))

        elif self.gnn_type == "GAT":
            from models.models_dgl import GATLayer
            num_heads = [4, 2, 1]
            activations = [F.elu, F.elu, None]
            self.gnn_layers = nn.ModuleList([GATLayer(in_channels, self.emb_size, num_heads=num_heads[0], activation=activations[0])])
            for i in range(args.layer_num_g - 1):
                self.gnn_layers.append(GATLayer(self.emb_size*num_heads[i], self.emb_size, num_heads=num_heads[i+1], activation=activations[i+1]))

        elif self.gnn_type == "GIN":
            from models.models_dgl import GINLayer, MLP
            activations = [F.elu, F.elu, None]
            mlp = MLP(in_channels, self.emb_size, self.emb_size)
            self.gnn_layers = nn.ModuleList([GINLayer(mlp, learn_eps=False, activation=activations[0])])
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.emb_size)])
            for i in range(args.layer_num_g - 1):
                mlp = MLP(self.emb_size, self.emb_size, self.emb_size)
                self.gnn_layers.append(GINLayer(mlp, learn_eps=False, activation=activations[i]))
                self.batch_norms.append(nn.BatchNorm1d(self.emb_size))


        self.pool = SumPooling()
    def forward(self, ob):
        ## Graph
        ob_g = [o['g'] for o in ob]
        ob_att = [o['att'] for o in ob]

        # create attachment point mask as one-hot
        for i, x_g in enumerate(ob_g):
            att_onehot = F.one_hot(torch.LongTensor(ob_att[i]), 
                        num_classes=x_g.number_of_nodes()).sum(0)
            ob_g[i].ndata['att_mask'] = att_onehot.bool()

        g = deepcopy(dgl.batch(ob_g)).to(self.device)
        
        g.ndata['x'] = self.emb_linear(g.ndata['x'])

        for i, conv in enumerate(self.gnn_layers):
            # h = conv(g, g.ndata['x'])
            h = conv(g)
            if self.gnn_type == "GAT":
                h = h.flatten(1)
            if self.gnn_type == "GIN":
                h = self.batch_norms[i](h)
            g.ndata['x'] = h

        
        emb_node = g.ndata['x']

        ## Get graph embedding
        emb_graph = self.pool(g, g.ndata['x'])
        
        return g, emb_node, emb_graph



