import sys
sys.path.append('/Users/vivianqwj/Downloads/retro_smiles_transformer-master/class_predictor/utils')
import rdkit.Chem as Chem
import torch.utils.data as data
# import units
import utils.data_utils as data_utils

import pdb

class RxnDataset(data.Dataset):
    def __init__(self, src_path, tgt_path, tgt_beam_size=1):
        # Assumes the data with reaction class labels

        def parse_func(line):
            splits = line.strip().split(' ')
            rxn_class_label = data_utils.parse_rxn_token(splits[0]) - 1
            smiles_tokens = splits[1:]
            return (rxn_class_label, ''.join(smiles_tokens))

        src_data = data_utils.read_file(src_path, parse_func=parse_func)
        classes, src_smiles = zip(*src_data)
        tgt_smiles = data_utils.read_file(tgt_path)

        # Expand the src input when the target has multiple solutions for each input
        if tgt_beam_size > 1:
            new_classes, new_src_smiles = [], []
            for idx in range(len(src_smiles)):
                new_classes += [classes[idx]] * tgt_beam_size
                new_src_smiles += [src_smiles[idx]] * tgt_beam_size
            classes = new_classes
            src_smiles = new_src_smiles

            # clean up targets, if smiles is invalid, replace with ''
            new_tgt_smiles = []
            for smiles in tgt_smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                except:
                    new_tgt_smiles.append('C')
                if mol is None:
                    new_tgt_smiles.append('C')
                else:
                    new_tgt_smiles.append(smiles)
            tgt_smiles = new_tgt_smiles
            
        self.data = list(zip(src_smiles, tgt_smiles, classes))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_loader(src_path, tgt_path, batch_size, tgt_beam_size=1,
               shuffle=False, num_workers=5):
    rxn_dataset = RxnDataset(
        src_path, tgt_path, tgt_beam_size=tgt_beam_size)

    data_loader = data.DataLoader(
        rxn_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return data_loader

#src_path = "../gnn_data/src_train.txt"
#tgt_path = "../gnn_data/tgt_train.txt"

# 这段代码的输入是两个文件路径：src_path和tgt_path。
# src_path是包含反应类标签和SMILES表示的文件，而tgt_path是只包含SMILES表示的文件。可以举一个src_data的例子：假如src_path对应的是一个文本文件，其中包括以下内容：
# 1 CCOCC(=O)NC1=CC=C(C=C1)C2=CC(=C(C=C2)Br)Cl
# 2 CC1(C)OC(C)C(NC(C)=O)C(O)C(C)O1
# 3 CC1(C)SC2C(CC3SC(C)(C)C(O)C(N)=N3)N(C)C(=O)C2(O)C1
#  那么，src_data将包含以下元素：
#  [(0, 'CCOCC(=O)NC1=CC=C(C=C1)C2=CC(=C(C=C2)Br)Cl'),
#  (1, 'CC1(C)OC(C)C(NC(C)=O)C(O)C(C)O1'),
#  (2, 'CC1(C)SC2C(CC3SC(C)(C)C(O)C(N)=N3)N(C)C(=O)C2(O)C1')]
#  其中，每个元素包含反应类标签和对应的SMILES表示。