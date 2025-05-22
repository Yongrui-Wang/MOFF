import rdkit
from rdkit import Chem
from rdkit.Chem import Lipinski, Descriptors,Crippen

def rule_of_five(smis):
      scores = []
      if len(smis) > 1:
            for smi in smis:
                  mol = Chem.MolFromSmiles(smi)
                  wm = Descriptors.MolWt(mol)
                  hd = Lipinski.NumHDonors(mol)
                  ha = Lipinski.NumHAcceptors(mol)
                  logp = Crippen.MolLogP(mol)
                  nrb = Lipinski.NumRotatableBonds(mol)
                  score = scoring(wm, hd, ha, logp, nrb)
                  scores.append(score)
      else:
            mol = Chem.MolFromSmiles(smis)
            wm = Descriptors.MolWt(mol)
            hd = Lipinski.NumHDonors(mol)
            ha = Lipinski.NumHAcceptors(mol)
            logp = Crippen.MolLogP(mol)
            nrb = Lipinski.NumRotatableBonds(mol)
            scores = scoring(wm, hd, ha, logp, nrb)
      # print('smi: {} \n Mol_Weight: {} \n HDonor: {} \n HAcceptors: {} \n SLogp: {} \n RotatableBonds: {} '
      #       .format(smi, wm, hd, ha, logp, nrb))
      return scores

def scoring(wm,hd,ha,logp,nrb):
      score = 0
      if 400<wm<500:
            score += 1
      if wm>620 or wm<300:
            score -= 2

      if hd<5:
            score += 1
      if hd>=5:
            score -= 2

      if ha <10:
            score += 1
      if ha >=10:
            score -= 2

      if logp <5:
            score += 1
      if logp >5:
            score -= 2

      if nrb <10:
            score += 1
      if nrb >=10:
            score -= 2

      return score

# score = rule_of_five(smi)
# print(score)