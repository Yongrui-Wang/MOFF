import sys
import os
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue


import subprocess
from openbabel import pybel

from rdkit import Chem
from rdkit.Chem import AllChem

from xml.etree import ElementTree as ET



class autodock_gpu(object):
    """
    autodock_gpu for covalent or non-covalent docking
    """

    def __init__(self, docking_params):
        """


        """
        super(autodock_gpu, self).__init__()

        self.name = docking_params['name']


        self.tmp_dir = docking_params['tmp_dir']
        self.maps_file = docking_params['receptor_maps']
        self.num_sub_proc = 3
        self.timeout_gen3d = docking_params['timeout_gen3d']
        self.timeout_dock = docking_params['timeout_dock']
        self.seed = docking_params['seed']

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def smi_2_pdbqt(self, smi, ligand_sdf_file, ligand_pdbqt_file):
        """
            generate initial 3d conformation from SMILES
            input :
                SMILES string
                ligand_mol_file (output file)
        """
        mol = AllChem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.MolToMolFile(mol,'{}'.format(ligand_sdf_file))
        run_line = 'mk_prepare_ligand.py -i {} -o {}'.format(ligand_sdf_file, ligand_pdbqt_file)

        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_gen3d, universal_newlines=True)



    def creator(self, q, data, num_sub_proc):
        """
            put data to queue
            input: queue
                data = [(idx1,smi1), (idx2,smi2), ...]
                num_sub_proc (for end signal)
        """
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking(self, ligand_pdbqt_file, ligand_xml_file):
        adgpu_run_line = './bin/autodock_gpu_64wi '
        adgpu_run_line += '--ffile {} '.format(self.maps_file)
        adgpu_run_line += '--lfile {} '.format(ligand_pdbqt_file)
        # adgpu_run_line += '--nrun {} '.format(8)
        # adgpu_run_line += '-a {} '.format(3)
        adgpu_run_line += '-x 1 '
        adgpu_run_line += '-s {} '.format(42)
        result = subprocess.check_output(adgpu_run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         universal_newlines=True)


        tree = ET.parse(r"{}".format(ligand_xml_file))
        root = tree.getroot()
        free_energy_o = root.find("runs").find("run").find('free_NRG_binding')
        free_energy = [free_energy_o.text]

        result_lines = result.split('\n')
        lis = result_lines[-13].split()
        affinity_list = [lis[4]]


        return free_energy


    def docking_sub(self, q, return_dict, sub_id=0):

        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            ligand_sdf_file = '{}/ligand_{}_{}.sdf'.format(self.tmp_dir, self.name, sub_id)
            ligand_pdbqt_file = '{}/ligand_{}_{}.pdbqt'.format(self.tmp_dir, self.name, sub_id)
            ligand_xml_file = '{}/ligand_{}_{}.xml'.format(self.tmp_dir, self.name, sub_id)
            try:
                self.smi_2_pdbqt(smi, ligand_sdf_file, ligand_pdbqt_file)
            except Exception as e:
                print(e)
                print("smi_2_pdbqt unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue


            try:
                affinity_list = self.docking( ligand_pdbqt_file,ligand_xml_file)
                # print(affinity_list)
            except Exception as e:
                print(e)
                print("docking unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue


            if len(affinity_list) == 0:
                affinity_list.append(99.9)

            affinity = affinity_list[0]
            return_dict[idx] = float(affinity)


    def dock(self, smiles):
        """

        """
        data = list(enumerate(smiles))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_m = Process(target=self.creator,
                              args=(q1, data, self.num_sub_proc))
        proc_m.start()
        proc_m.join()

        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_sub,
                           args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        # proc_m.join()
        for proc in procs:
            proc.join()
        keys = sorted(return_dict.keys())
        affinity_list = list()
        for key in keys:
            affinity = return_dict[key]
            affinity_list += [affinity]
        return affinity_list


