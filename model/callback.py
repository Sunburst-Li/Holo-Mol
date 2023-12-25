from model.loss import *
import torch as K
import torch
from torch.nn import functional as F
from tqdm import tqdm
import operator
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.test = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, test, n=0):
        self.test = test
        self.sum += test * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return str(self.test)
        return '%.4f (%.4f)' % (self.test, self.avg)


class Roc():
    def __init__(self, test_gen, model, hyper=None):
        super(Roc, self).__init__()
        self.model = model
        self.test_gen = test_gen
        self.hyper = hyper

    def on_epoch_end(self, epoch, train_mode=None):
        test_mse, test_rmse, test_loss = calculate_roc_pr(
                                                        model=self.model, 
                                                       sequence=self.test_gen,
                                                       epoch=epoch,
                                                       hyper=self.hyper,
                                                        train_mode=train_mode,
                                                      )
        return test_rmse

def calculate_roc_pr(model, sequence, epoch=0, hyper=None, train_mode=None):
    device = torch.device(hyper.CUDA if torch.cuda.is_available() else "cpu")
    path = hyper.root_path + "{}/{}/{}_layer{}_head{}_{}/fold_{}".format(hyper.dataset,
                                                                      hyper.model,
                                                                      hyper.dataset,
                                                                      str(hyper.num_layers),
                                                                      str(hyper.nheads),
                                                                      str(hyper.units_conv),
                                                                      str(hyper.fold)
                                                                   )
    y_pred = []
    y_true = sequence.y
    simles = sequence.s
    model.eval()
    test_loss = 0
    print("data_std:", hyper.data_std)
    print("data_mean:",hyper.data_mean)
    if train_mode == None:
        print("=== Epoch %d test ===" % epoch)
    else:
        print()
        print("*"*150)
        print("=== flod " + str(epoch) + " " + train_mode + " ===")
    avg_sent_loss = AverageMeter()
    t = tqdm(sequence)
    for iter, datas in enumerate(t):
        data_temp, batch_y,batch_s = datas
        indexs = batch_s
        if len(batch_y) == len(batch_s):
            batch_s = sequence.s[batch_s]
        else:
            batch_s = range(batch_s[0],(batch_s[0]) + len(batch_y))
            batch_s = sequence.s[batch_s]
        atom_tensor, adjm_tensor, posn_tensor = data_temp
        if int(atom_tensor.shape[0]) == 0:
            break
        batch_y = batch_y.to(device)
        atom_tensor = atom_tensor.to(device)
        adjm_tensor = adjm_tensor.to(device)
        posn_tensor = posn_tensor.to(device)
        batch_s = np.array(batch_s)
        input = model([atom_tensor, adjm_tensor, posn_tensor,batch_s])
        input = input.squeeze()
        try:
            loss = std_mse(hyper.data_std)(input.squeeze(), batch_y)
        except:
            loss = std_mse(hyper.data_std)(input.squeeze().to(device), batch_y.to(device))
        try:
            y_pred = y_pred + input.tolist()
        except:
            y_pred.append(input.item())
        avg_sent_loss.update(loss.item(), 1)
        t.set_postfix(sent_loss=avg_sent_loss.avg)
        test_loss = avg_sent_loss.__str__().split(' (')[1].split(')')[0]
    y_pred = torch.as_tensor(np.array(y_pred)).float()
    y_true = torch.as_tensor(np.array(y_true)).float()
    test_rmse = std_rmse(hyper.data_std)(y_pred, y_true)
    test_mse = std_mse(hyper.data_std)(y_pred, y_true)
    test_mae = std_mae(hyper.data_std)(y_pred, y_true)
    test_r2_score = r2_s(y_pred, y_true)
    test_r2 = std_r2(hyper.data_std)(y_pred, y_true)
    if train_mode == 'train':
        print()
        with open(path + r"/flod_train_lxb_mse.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_mse) + "\n")
        with open(path + r"/flod_train_lxb_mae.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_mae) + "\n")
        with open(path + r"/flod_train_lxb_r2.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_r2) + "\n")
        with open(path + r"/flod_train_lxb_rmse.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_rmse) + "\n")
        with open(path + r"/flod_train_lxb_r2_score.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_r2_score) + "\n")
    elif train_mode == 'test':
        with open(path + r"/flod_test_lxb_mse.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_mse) + "\n")
        with open(path + r"/flod_test_lxb_mae.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_mae) + "\n")
        with open(path + r"/flod_test_lxb_r2.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_r2) + "\n")
        with open(path + r"/flod_test_lxb_rmse.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_rmse) + "\n")
        with open(path + r"/flod_test_lxb_r2_score.txt", 'a', encoding='utf-8') as file:
            file.write(str(test_r2_score) + "\n")
    return test_mse, test_rmse, test_loss
    
    
