# Improving Covalent and Non-Covalent Molecule Generation via Reinforcement Learning with Functional Fragments

## Abstract

Small-molecule drugs play a critical role in cancer therapy by selectively targeting key signaling pathways that drive tumor growth. While deep learning models have advanced drug discovery, there remains a lack of generative frameworks for \textit{de novo} covalent molecule design using a fragment-based approach. To address this, we propose MOFF (MOlecule generation with Functional Fragments), a reinforcement learning framework for molecule generation. MOFF is specifically designed to generate both covalent and non-covalent compounds based on functional fragments. The model leverages docking scores as reward function and is trained using the Soft Actor-Critic algorithm. We evaluate MOFF through case studies targeting Bruton's tyrosine kinase (BTK) and the epidermal growth factor receptor (EGFR), demonstrating that MOFF can generate ligand-like molecules with favorable docking scores and drug-like properties, compared to baseline models and ChEMBL compounds. As a computational validation, molecular dynamics (MD) simulations were conducted on selected top-scoring molecules to assess potential binding stability. These results highlight MOFF as a flexible and extensible framework for fragment-based molecule generation, with the potential to support downstream applications.

---

## Environment Setup

Install required dependencies using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate molecule-rl
```
This project also requires AutoDock-GPU for docking. Follow the [AutoDock-GPU installation guide](https://github.com/ccsb-scripps/AutoDock-GPU/wiki/Guideline-for-users) to compile it. Once built, add the binary to the `./bin` directory.

## Running Molecular Generation
We provide two example shell scripts for covalent and non-covalent molecule generation using reinforcement learning guided by docking scores.

### Covalent Molecule Generation (Target: BTK):
Save as `run_cov.sh`:
```bash
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
```

### Non-Covalent Molecule Generation (Target: BTK)
Save as `run_noncov.sh`:
```bash
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
```
### Run with EGFR Target
covalent:
```
--receptor_pdb='gym_molecule/maps_file/2j5f_cov/2j5f_protein.pdb' \
--covlent_amino_acid='A:CYS:797' \
--receptor_maps='gym_molecule/maps_file/2j5f_cov/2j5f_protein_rigid.maps.fld'
```
non-covalent:
```
--receptor_maps='gym_molecule/maps_file/2j5f/2j5f_protein.maps.fld'
```


## Extendibility
Fragments used to build molecules are stored in:
```
./gym_molecule/dataset/*.txt
```
Each text file corresponds to a functional group type (e.g., warhead, linker, etc.).

You can change the build logic using:

```
--step_list 0 2 1 2
```

This represents the order in which fragment types are assembled during molecule generation. The numbers correspond to indices of your custom fragment categories.


Extend to **new protein receptors**:

1. Prepare your receptor .pdb and generate docking .maps.fld files.

2. Follow AutoDock-Vina or AutoDock-GPU documentation:

    * [Basic docking](https://autodock-vina.readthedocs.io/en/latest/docking_basic.html)

    * [Covalent docking](https://autodock-vina.readthedocs.io/en/latest/docking_flexible.html)

3. Replace `--receptor_maps`, and optionally `--receptor_pdb`, `--covlent_amino_acid` in the script.
