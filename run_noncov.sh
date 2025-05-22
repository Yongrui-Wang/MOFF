export PATH="bin:$PATH"  # for docking use

CUDA_LAUNCH_BLOCKING=1 python3 run_rl.py \
                       --name='n1_8848' \
                       --load=0 --train=1 --has_feature=1 \
                       --name_full_load='' \
                       --min_action=1 --max_action=4 \
                       --gnn_aggregate='sum' --gnn_type='GCN'\
                       --seed=8848 --intr_rew=0 --intr_rew_ratio=5e-1  \
                       --update_after=3000 --start_steps=4000 --update_every=256 --init_alpha=1. \
                       --is_covalent=0 --lipinski_rew=0\
                       --desc='ecfp' \
                       --rl_model='sac' \
                       --active_learning='moff' \
                       --gpu_id=0 --emb_size=96 --tau=.1 --batch_size=256 --target_entropy=0.1\
                       --munchausen=1 --alpha_min=0.1 --init_alpha_lr=5e-4\
                       --step_list 0 2 1 2\
                       --receptor_maps='gym_molecule/maps_file/6e4f/6e4f_protein.maps.fld'