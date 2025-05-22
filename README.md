# Improving Covalent and Non-Covalent Molecule Generation via Reinforcement Learning with Functional Fragments

## 📝 Abstract

<!-- Paste your paper's abstract below -->
*To be added by the author...*

---

## 🧪 Environment Setup

Install required dependencies using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate molecule-rl
This project also requires AutoDock-GPU for docking. Follow the AutoDock-GPU installation guide to compile it. Once built, add the binary to your PATH:

bash
复制
编辑
export PATH="bin:$PATH"
🚀 Running Molecular Generation
We provide two example shell scripts for covalent and non-covalent molecule generation using reinforcement learning guided by docking scores.

🔗 Covalent Generation
bash
复制
编辑
export PATH="bin:$PATH"  # for docking use

CUDA_LAUNCH_BLOCKING=1 python3 run_rl.py \
    --name='c1_3267' \
    --load=0 --train=1 --has_feature=1 \
    --name_full_load='' \
    --min_action=1 --max_action=4 \
    --gnn_aggregate='sum' --gnn_type='GCN'\
    --seed=3267 --intr_rew=0 --intr_rew_ratio=5e-1  \
    --update_after=3000 --start_steps=4000 --update_every=256 --init_alpha=1. \
    --is_covalent=1 --lipinski_rew=0 \
    --desc='ecfp' \
    --rl_model='sac' \
    --active_learning='moff' \
    --gpu_id=0 --emb_size=96 --tau=.1 --batch_size=256 --target_entropy=0.1 \
    --munchausen=1 --alpha_min=0.1 --init_alpha_lr=5e-4 \
    --step_list 0 3 1 2 \
    --receptor_pdb='gym_molecule/maps_file/5p9j/5p9j.pdb' \
    --covlent_amino_acid='A:CYS:481' \
    --receptor_maps='gym_molecule/maps_file/5p9j/5p9j_rigid.maps.fld'
🔗 Non-Covalent Generation
bash
复制
编辑
export PATH="bin:$PATH"  # for docking use

CUDA_LAUNCH_BLOCKING=1 python3 run_rl.py \
    --name='n1_8848' \
    --load=0 --train=1 --has_feature=1 \
    --name_full_load='' \
    --min_action=1 --max_action=4 \
    --gnn_aggregate='sum' --gnn_type='GCN'\
    --seed=8848 --intr_rew=0 --intr_rew_ratio=5e-1  \
    --update_after=3000 --start_steps=4000 --update_every=256 --init_alpha=1. \
    --is_covalent=0 --lipinski_rew=0 \
    --desc='ecfp' \
    --rl_model='sac' \
    --active_learning='moff' \
    --gpu_id=0 --emb_size=96 --tau=.1 --batch_size=256 --target_entropy=0.1 \
    --munchausen=1 --alpha_min=0.1 --init_alpha_lr=5e-4 \
    --step_list 0 2 1 2 \
    --receptor_maps='gym_molecule/maps_file/6e4f/6e4f_protein.maps.fld'
📂 Receptor Map Files
You can use the following receptor .maps.fld files:

Target	Covalent	Non-Covalent
BTK	gym_molecule/maps_file/5p9j/5p9j_rigid.maps.fld	—
EGFR	—	gym_molecule/maps_file/6e4f/6e4f_protein.maps.fld

For covalent generation, ensure you also provide:

The .pdb file: --receptor_pdb='...'

The covalent binding residue: --covlent_amino_acid='A:CYS:481'

🧩 Customize Fragments and Construction Logic
Fragments used to build molecules are stored in:

bash
复制
编辑
MOFF/gym_molecule/dataset/*.txt
Each text file corresponds to a functional group type (e.g., warhead, linker, etc.).

You can change the build logic using:

bash
复制
编辑
--step_list 0 2 1 2
This represents the order in which fragment types are assembled during molecule generation. The numbers correspond to indices of your custom fragment categories.

🧬 Extend to Other Protein Targets
To apply this framework to new proteins:

Prepare your receptor .pdb and generate docking .maps.fld files.

Follow AutoDock-Vina or AutoDock-GPU documentation:

Basic docking

Flexible docking

Replace --receptor_pdb, --receptor_maps, and optionally --covlent_amino_acid in the script.

📫 Contact
For any questions, feel free to reach out:

📧 Yongrui Wang: wangyongrui20@mails.ucas.ac.cn

📧 Xiaolin Li: xiaolinli@ieee.org
