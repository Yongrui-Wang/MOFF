#!/usr/bin/env python3

import os
import random

import numpy as np
import dgl
import torch 
from tensorboardX import SummaryWriter

import gym

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)

def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda:"+str(gpu_id))
    else:
        print('cuda not available')
        device = torch.device("cpu")

    return device

def train(args,seed,writer=None):

    from moff import sac

    from models.core_motif import GNNActorCritic

    workerseed = args.seed
    set_seed(workerseed)
    
    # device
    gpu_use = False
    gpu_id = None
    if args.gpu_id is not None:
        gpu_id = int(args.gpu_id)
        gpu_use = True

    device = gpu_setup(gpu_use, gpu_id)

    env = gym.make('molecule-v0')
    env.init(docking_config=args.docking_config, ratios = args.ratios, reward_step_total=args.reward_step_total,
             is_normalize=args.normalize_adj,has_feature=bool(args.has_feature),max_action=args.max_action,
             min_action=args.min_action,is_covalent=bool(args.is_covalent))
    env.seed(workerseed)


    SAC = sac(writer, args, env, actor_critic=GNNActorCritic, ac_kwargs=dict(), seed=seed,
        steps_per_epoch=500, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=args.init_lr, alpha=args.init_alpha, batch_size=args.batch_size, start_steps=args.start_steps,
        update_after=args.update_after, update_every=args.update_every, update_freq=args.update_freq,
        expert_every=5, num_test_episodes=8, max_ep_len=args.max_action,
        save_freq=args.save_every, train_alpha=True)
    SAC.train()

    env.close()

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def molecule_arg_parser():
    parser = arg_parser()


    parser.add_argument('--rl_model', type=str, default='sac') # RL模型选择

    parser.add_argument('--gpu_id', type=int, default=0) #None -> 0
    parser.add_argument('--train', type=int, default=1, help='training or inference')#
    # env
    parser.add_argument('--env', type=str, help='environment name: molecule; graph', default='molecule')
    parser.add_argument('--seed', help='work seed', type=int, default=42)
    parser.add_argument('--num_steps', type=int, default=int(5e7))

    parser.add_argument('--name',type=str,default='work_name') # name
    parser.add_argument('--name_full',type=str,default='') # save name
    parser.add_argument('--name_full_load',type=str,default='')
    

    parser.add_argument('--reward_step_total', type=float, default=0.5)

    parser.add_argument('--step_list', type=int, nargs='+', default=[0,3,1,2]) # noncov:0,2,1,2  cov:0,3,1,2
    parser.add_argument('--is_covalent', type=int, default=1, help='is for covalent docking?')
    parser.add_argument('--receptor_pdb',type=str, default='gym_molecule/maps_file/5p9j/5p9j.pdb')
    parser.add_argument('--receptor_maps', type=str, default='gym_molecule/maps_file/5p9j/5p9j_rigid.maps.fld')
    parser.add_argument('--covlent_amino_acid', type=str, default='A:CYS:481',
                        help='covalent docking amino acid and its chain, such as A:CYS:481')

    parser.add_argument('--lipinski_rew', type=int, default=0,
                        help='using Lipinski\'s rule of five as one part of reward' )


    parser.add_argument('--intr_rew', type=str, default=None) # intr, mc
    parser.add_argument('--intr_rew_ratio', type=float, default=5e-1)
    
    parser.add_argument('--tau', type=float, default=1e-1)


    # model update
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--update_every', type=int, default=256)
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_after', type=int, default=2000)
    parser.add_argument('--start_steps', type=int, default=3000)
    
    # model save and load
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=250)
    
    # graph embedding
    parser.add_argument('--gnn_type', type=str, default='CCN')
    parser.add_argument('--gnn_aggregate', type=str, default='sum')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--has_residual', type=int, default=0)
    parser.add_argument('--has_feature', type=int, default=1)

    parser.add_argument('--normalize_adj', type=int, default=0)
    parser.add_argument('--bn', type=int, default=0)

    parser.add_argument('--layer_num_g', type=int, default=3)

    parser.add_argument('--max_action', type=int, default=4) 
    parser.add_argument('--min_action', type=int, default=1) 

    # SAC
    parser.add_argument('--target_entropy', type=float, default=1.)
    parser.add_argument('--init_alpha', type=float, default=1.)
    parser.add_argument('--desc', type=str, default='ecfp') # ecfp
    parser.add_argument('--init_pi_lr', type=float, default=1e-4)
    parser.add_argument('--init_q_lr', type=float, default=1e-5)
    parser.add_argument('--init_alpha_lr', type=float, default=1e-4)
    parser.add_argument('--init_p_lr', type=float, default=3e-4)
    parser.add_argument('--alpha_max', type=float, default=20.)
    parser.add_argument('--alpha_min', type=float, default=0.05)

    # MC dropout
    parser.add_argument('--active_learning', type=str, default='moff')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n_samples', type=int, default=5)

    parser.add_argument("--munchausen", type=int, default=1, choices=[0, 1],
                        help="Adding Munchausen RL to the agent if set to 1, default = 0")
    parser.add_argument("--ere", type=int, default=0, choices=[0, 1],
                        help="Adding Emphasizing Recent Experience to the agent if set to 1, default = 0")
    # On-policy
    parser.add_argument('--n_cpus', type=int, default=1) #
    parser.add_argument('--steps_per_epoch', type=int, default=257) #



    return parser

def main():
    args = molecule_arg_parser().parse_args()
    print(args)
    args.name_full = args.env + '_' + args.name

    docking_config = dict()


    docking_config['docking_program'] = 'bin/autodock_gpu_64wi'
    docking_config['tmp_dir'] = 'tmp'
    docking_config['receptor_pdb'] = args.receptor_pdb
    docking_config['rec_residue'] = args.covlent_amino_acid
    docking_config['receptor_maps'] = args.receptor_maps
    docking_config['name'] = args.name_full

    docking_config['n_process'] = 2
    docking_config['timeout_gen3d'] = 30
    docking_config['timeout_dock'] = 100 
    docking_config['seed'] = args.seed

    ratios = dict()
    ratios['logp'] = 0
    ratios['qed'] = 0
    ratios['sa'] = 0
    ratios['mw'] = 0
    ratios['filter'] = 0
    ratios['docking'] = 1

    args.docking_config = docking_config
    args.ratios = ratios
    
    # check and clean
    if not os.path.exists('molecule_gen'):
        os.makedirs('molecule_gen')
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    writer = SummaryWriter(comment='_'+args.name)

    # device
    gpu_use = False
    gpu_id = None
    if args.gpu_id is not None:
        gpu_id = int(args.gpu_id)
        gpu_use = True
    device = gpu_setup(gpu_use, gpu_id)
    args.device = device

    if args.gpu_id is None:
        torch.set_num_threads(256)
        print(torch.get_num_threads())

    train(args,seed=args.seed,writer=writer)

if __name__ == '__main__':
    main()

