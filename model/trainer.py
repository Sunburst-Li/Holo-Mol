from model.dataset import Dataset
from model import model as m
from model.callback import *
from datetime import datetime
from model.loss import *
import numpy as np
import torch
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from model import pubchemfp
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

class EarlyStopping():
    def __init__(self, model=None, patience=5, min_delta=0, restore_best_weights=True,model_path=None):
        self.model = model
        self.model_path = model_path
        self.restore_best_weights = restore_best_weights
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, test_loss):
        if self.best_loss == None:
            self.best_loss = test_loss
        elif self.best_loss - test_loss > self.min_delta:
            self.best_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class LRScheduler():
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5, eps=1e-8, verbose=True
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.eps = eps
        self.verbose = verbose
        self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.epoch = 0
        self.num_bad_epochs = 0
        self.lr = []
        self.best = None
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.lr.append(float(param_group['lr']))
    def __call__(self, test_loss):
        if self.best == None:
            self.best = test_loss
        elif test_loss - self.best > 0 :
            self.num_bad_epochs = self.num_bad_epochs + 1
        else:
            self.best = test_loss
            self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    param_group['lr'] = new_lr
                    if self.verbose:
                        print('Epoch {:5d}: reducing learning rate'
                              ' of group {} to {:.4e}.'.format(self.epoch, i, new_lr))
                else:
                    param_group['lr'] = self.lr[i]
        self.epoch = self.epoch + 1
        
def l1_regularization(model, l1_alpha):
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            module.weight.grad.data.add_(l1_alpha * torch.sign(module.weight.data))

def l2_regularization(model, l2_alpha=1e-3):
    for module in model.modules():
        alpha = torch.sum(torch.square(module.weight.data)) *l2_alpha
        module.weight.grad.data.add_(module.weight.data * alpha)

class Trainer(object):
    def __init__(self, config):
        self.data = None
        self.model = None
        self.hyper = config

    def __repr__(self):
        text = ""
        for key, value in self.log.items():
            text += "{}:\t".format(key)
            for error in value[0]:
                text += "{0:.4f} ".format(float(error))
            text += "\n"
        return text

    def load_data(self, batch=128,fold=1):
        self.data = Dataset(self.hyper.dataset, batch=batch, fold=fold)

    def load_model(self, ):
        self.model = getattr(m , self.hyper.model_class)(self.hyper)

    def fit(self):
        device=torch.device(self.hyper.CUDA if torch.cuda.is_available() else "cpu")
        self.load_data(batch=self.hyper.batch, fold=self.hyper.fold_total)
        for i in range(1, self.hyper.fold_total + 1):
            self.hyper.fold = i
            path = self.hyper.root_path + "{}/{}/{}_layer{}_head{}_{}/fold_{}".format(self.hyper.dataset,
                                                                                    self.hyper.model,
                                                                                    self.hyper.dataset, 
                                                                                   str(self.hyper.num_layers), 
                                                                                   str(self.hyper.nheads), 
                                                                                   str(self.hyper.units_conv),
                                                                                   str(self.hyper.fold)
                                                                                  )
            model_path = path + r"/{}/".format(self.hyper.model)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(path+ r"/epoch"):
                os.makedirs(path + r"/epoch")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            asses = 1000000000000000
            self.data.mean = np.mean(self.data.y_original["train"][self.hyper.fold-1])
            self.data.std = np.std(self.data.y_original["train"][self.hyper.fold-1])
            print()
            print("mean:",self.data.mean)
            print("std:",self.data.std)
            self.data.y["train"] = (self.data.y_original["train"][self.hyper.fold-1] - self.data.mean) / self.data.std
            self.data.y["test"] = (self.data.y_original["test"][self.hyper.fold-1] - self.data.mean) / self.data.std
            self.data.x["train"] = self.data.x_original["train"][self.hyper.fold-1]
            self.data.x["test"] = self.data.x_original["test"][self.hyper.fold-1]
            self.data.c["train"] = self.data.c_original["train"][self.hyper.fold-1]
            self.data.c["test"] = self.data.c_original["test"][self.hyper.fold-1]
            self.data.s["train"] = self.data.s_original["train"][self.hyper.fold-1]
            self.data.s["test"] = self.data.s_original["test"][self.hyper.fold-1]
            self.hyper.num_train = len(self.data.y["train"])
            self.hyper.num_test = len(self.data.y["test"])
            self.hyper.num_atoms = self.data.max_atoms
            self.hyper.num_features = self.data.num_features
            self.hyper.data_std = self.data.std
            self.hyper.data_mean = self.data.mean
            self.hyper.task = self.data.task
            self.data.set_features(self.hyper.features)
            self.hyper.features["num_features"] = self.data.num_features
            self.load_model()
            self.model.to(device)
            print(self.model)
            callbacks = Roc(self.data.generator("test"), self.model, hyper=self.hyper)
            optimizer = optim.Adam(self.model.parameters(), lr=self.hyper.lr)
            lrs = LRScheduler(optimizer, self.hyper.patience, self.hyper.min_lr, self.hyper.factor)
            early_stopping = EarlyStopping(self.model, self.hyper.stop_patience, model_path=model_path)
            print(self.model)
            data_epoch = self.data.generator("train")
            for epo in range(self.hyper.epoch):
                self.model.train()
                print("=== Epoch %d train ===" % epo)
                avg_sent_loss = AverageMeter()
                t = tqdm(data_epoch)
                lens = len(data_epoch)
                avg_loss = []
                for iter, datas in enumerate(t):
                    data_temp, batch_y, batch_s = datas
                    if len(batch_y) == len(batch_s):
                        batch_s = data_epoch.s[batch_s]
                    else:
                        batch_s = range(batch_s[0],(batch_s[0]) + len(batch_y))
                        batch_s = data_epoch.s[batch_s]
                    atom_tensor, adjm_tensor, posn_tensor = data_temp
                    if int(atom_tensor.shape[0]) == 0:
                        break
                    batch_y = batch_y.to(device)
                    atom_tensor = atom_tensor.to(device)
                    adjm_tensor = adjm_tensor.to(device)
                    posn_tensor = posn_tensor.to(device)
                    batch_s = np.array(batch_s)  
                    input = self.model([atom_tensor, adjm_tensor, posn_tensor, batch_s])
                    loss = torch.mean(torch.square(input.squeeze() - batch_y))
                    optimizer.zero_grad()
                    loss.backward()
                    #  l2_regularization(self.model.sc_out1, self.hyper.l2_parm)
                    #  l2_regularization(self.model.sc_out2, self.hyper.l2_parm)
                    #  l2_regularization(self.model.vc_out1, self.hyper.l2_parm)
                    #  l2_regularization(self.model.vc_out2, self.hyper.l2_parm)
                    optimizer.step()
                    avg_sent_loss.update(loss.item(), 1)
                    t.set_postfix(sent_loss=avg_sent_loss.avg)
                    avg_loss.append(avg_sent_loss.avg)
                    with open(path + r"/train_loss.txt", 'a', encoding='utf-8') as file:
                        file.write(str(avg_sent_loss.avg) + "\n")
                avg_loss = sum(avg_loss) / len(avg_loss)
                with open(path + r"/epoch_train_loss.txt", 'a', encoding='utf-8') as file:
                    file.write(str(avg_loss) + "\n")
                if epo % self.hyper.test_print == 0:
                    test_mse = callbacks.on_epoch_end(epo)
                    lrs(test_mse.item())
                    torch.save(self.model.state_dict(), model_path + 'model.pth')
                    if asses > test_mse.item():
                        asses = test_mse.item()
                        torch.save(self.model.state_dict(), model_path + 'best_model.pth')
                    early_stopping(float(test_mse.item()))
                    if early_stopping.early_stop:
                        break
            print()
            print("========================best_model.pth===========================")
            self.hyper.model_type = "best_model"
            for gen in ["test","train"]:
                state_dict = torch.load(model_path + 'best_model.pth')
                self.model.load_state_dict(state_dict)
                roc = Roc(self.data.generator(gen), self.model, hyper=self.hyper)
                test_rmse = roc.on_epoch_end(self.hyper.fold,train_mode=gen)
            print()
            print("========================model.pth===========================")
            self.hyper.model_type = "model"
            for gen in ["test","train"]:
                state_dict = torch.load(model_path + 'model.pth')
                self.model.load_state_dict(state_dict)
                roc = Roc(self.data.generator(gen), self.model, hyper=self.hyper)
                test_rmse = roc.on_epoch_end(self.hyper.fold,train_mode=gen)
        print("Training Ended")
