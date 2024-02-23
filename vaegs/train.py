import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .utils import *
from .model import *
import shap

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).float()
        if len(self.label.shape) == 1:
            self.label = self.label.unsqueeze(1)
        self.length = label.shape[0]
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return self.length

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def evaluate_file(net, test_dataset, file_name, device=None, is_vae=True, prefix = ''):
    if isinstance(net, nn.Module):
        net.eval()  
        if not device:
            device = next(iter(net.parameters())).device
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    pres = []
    labels = []
    for batch_idx, (data, label) in enumerate(test_dataloader):
        # print("Iteration %d: " % batch_idx, data.shape, labels.shape)
        data = data.to(device)
        # labels = labels.to(device)
        pre = net(data)
        # if is_vae:
        #     pre =pre[0]
        pres.append(pre.detach().cpu())
        labels.append(label)
    y_tests, y_predicts = torch.cat(labels),torch.cat(pres)
    print(y_tests.shape, y_predicts.shape)
    for i in range(y_tests.shape[1]):
        result = [] 
        y_test, y_predict = y_tests[:,i], y_predicts[:,i]
        result.append(prefix)
        result.append(str(i))
        result.append(str(mean_absolute_error(y_test, y_predict)))
        result.append(str(np.corrcoef(y_test, y_predict)[0][1]))
        print(result)
        appendresult([','.join(result)], file_name)


        
def train_multi_kl(net, train_data, device, is_vae = True, vae_ratio=0.5, num_epochs=10, 
                   lr=1e-5, batch_size=32, shuffle=True, num_workers=0, print_skip = 10, opt = 'adam', init=True):
    if init:
        net.apply(init_weights)
    print('training on', device, str(init))
    net.to(device)
    if opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.MSELoss()
    for epoch in range(num_epochs):
        # print('epoch:', epoch)
        net.train()
        el1 = 0
        el2 = 0
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            out = net(X)            
            if is_vae:
                y_hat = out[0]
                X_hat = out[1]
                l1 = 0
                for i in range(y.shape[1]):
                    y_hat = out[0][:,i]
                    yi = y[:,i]
                    l1 += loss(y_hat, yi)
                el1 += l1
                
                mu, logvar = out[2],out[3]
                kl_loss = -0.5*torch.mean(logvar+1-mu**2-torch.exp(logvar))   
                # print(kl_loss)
                l2 = loss(X_hat, X) + kl_loss
                el2 += l2
                l = (1-vae_ratio)*l1 + vae_ratio*l2
            else:
                l = 0
                for i in range(y.shape[1]):
                    y_hat = out[0][:,i]
                    yi = y[:,i]
                    l += loss(y_hat, yi)
                mu, logvar = out[1],out[2]
                kl_loss = -0.5*torch.mean(logvar+1-mu**2-torch.exp(logvar))  
                l += kl_loss
                el1 += l
            l.backward()
            optimizer.step()
        if epoch==0 or (epoch+1) % print_skip ==0:
            print('epoch %d loss:'%(epoch+1), el1/(batch_idx+1),el2/(batch_idx+1))
            
def train_test_klcv(net, data, label, fold =5, is_vae = True, num_epochs=50, lr=1e-4, batch_size=32, vae_ratio=0.5, print_skip=10,device=None,sample_size = None, opt = 'adam', file_name =None, seed = 42, init=True):
    mydata = MyDataSet(data, label)
    size_list = [1.0/fold]*fold
    data_list = random_split(mydata, size_list,torch.Generator().manual_seed(seed))
    if file_name is None:
        file_name = 'result.csv'

    # record net and parameters
    result = [type(net)]   
    result.append(num_epochs)
    result.append(lr)
    result.append(batch_size)
    result.append(vae_ratio)
    result.append(sample_size)
    result.append(opt)
    print(result)
    result = list(map(lambda x:str(x),result))        
    appendresult([','.join(result)], file_name)
    
    for k in range(fold):
        print('fold ', k)
        train_dataset = None
        for i in range(fold):
            if i ==k:
                test_dataset = data_list[i]
            else:
                if train_dataset is None:
                    train_dataset = data_list[i]
                else:
                    train_dataset += data_list[i]
    
        if sample_size is not None:
            np.random.seed(seed)
            indicies = np.random.randint(len(train_dataset), size=sample_size)
            train_dataset = torch.utils.data.Subset(train_dataset,indicies)
            
        prefix = str(k)
        train_multi_kl(net, train_dataset, device, is_vae = is_vae, num_epochs=num_epochs, lr=lr, 
                    batch_size=batch_size, vae_ratio=vae_ratio, print_skip=print_skip, opt = opt,init=init)
        evaluate_file(net, test_dataset, file_name, device, True, prefix)
        
def get_topindex_net(X_train, y_train, device, n=1000):    
    train_dataset = MyDataSet(X_train, y_train)
    net = VMGP(X_train.shape[1], out_dim=y_train.shape[1])
    train_multi_kl(net, train_dataset, device, num_epochs=50,
                lr=0.0001, print_skip = 50, batch_size=32)
    xt = torch.from_numpy(X_train).float()
    xt = xt.to(device)
    explainer = shap.DeepExplainer(net,xt)
    
    shap_values_all = np.array(explainer.shap_values(xt))
    mean_shape = np.mean(np.mean(shap_values_all,axis=0),axis=0)
    sorted_index_array = mean_shape.argsort()
    return sorted_index_array[-n:]