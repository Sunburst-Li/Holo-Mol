import sys
sys.path.append(r'/root/Holo-Mol')
import sys
import os
import torch
import random, argparse
from model.trainer import Trainer
from configs import Config
import warnings
warnings.filterwarnings("ignore")
torch.cuda.set_device(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def seed_torch(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    features = {"use_atom_symbol": True, "use_degree": True, "use_hybridization": True, "use_implicit_valence": True,
                "use_partial_charge": True, "use_ring_size": True, "use_hydrogen_bonding": True,
                "use_acid_base": True, "use_aromaticity": True, "use_chirality": True, "use_num_hydrogen": True}
    
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--batch', default=20, type=int)
    parser.add_argument('--fold_total', default=10, type=int)
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--units_conv', default=32, type=int)
    parser.add_argument('--units_dense', default=32, type=int)
    parser.add_argument('--outputs', default=1, type=int)
    parser.add_argument('--dataset', default='group', type=str)
    parser.add_argument('--pooling', default='sum', type=str)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--l2_parm', default=0.005, type=float)
    parser.add_argument('--CUDA', default="cuda", type=str)
    parser.add_argument('--test_print', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--nheads', default=1, type=int)
    parser.add_argument('--stop_patience', default=20, type=int)
    parser.add_argument('--features', default=features, type=dict)
    parser.add_argument('--root_path', default="/root/Holo-Mol/", type=str)
    parser.add_argument('--model_class', default="Holo_Mol", type=str)
    parser.add_argument('--model', default="Holo_Mol", type=str)
    args = parser.parse_args()
    config = Config(args)
    trainer = Trainer(config)
    trainer.fit()
