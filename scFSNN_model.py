from torch import nn
import numpy as np
import math
import random
from sklearn import preprocessing
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from scipy import stats
from sklearn.model_selection import train_test_split


def aug_train_alpha(expr, labels, Fold=1, d=0.2):

    n_cell, n_gene = expr.shape
    expr1 = np.tile(expr,(Fold,1)) 
    expr2 = expr1.copy()
    np.random.shuffle(expr2)
    
    alpha = (np.random.rand(n_cell) * d).reshape([-1, 1])
    expr1 = expr1 * (1-alpha) + expr2 * alpha
    
    aug_train = np.vstack((expr, expr1)) 
    aug_labels = np.tile(labels, (Fold+1,1))
    
    return aug_train, aug_labels


def pre_process(x, y, d=0.2, test_prob=0.2, val_prob=0.2, Fold = 1):
    
    # split data into train, test datasets
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_prob)
    # split train dataset into train, validation datasets
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_prob)
    
    # normalize data to make the data have same library sizes
    train_library_size = np.sum(train_x, axis=1, keepdims=True)
    median_library_size = np.median(train_library_size)
    
    print("median_library_size:", median_library_size)
    
    train_x_normed = train_x/train_library_size * median_library_size
    
    test_library_size = np.sum(test_x ,axis=1, keepdims=True)
    test_x_normed = test_x/test_library_size * median_library_size
    
    val_library_size = np.sum(val_x, axis=1, keepdims=True)
    val_x_normed = val_x/val_library_size * median_library_size
    
    # augument data
    if Fold is not None:
        train_x_normed, train_y = aug_train_alpha(train_x_normed, train_y, Fold=Fold, d=d)
    
    # take logarithm
    train_x_log = np.log1p(train_x_normed)
    val_x_log = np.log1p(val_x_normed)
    test_x_log = np.log1p(test_x_normed)
    
    # normalize data to make the data have means 0 and variance 1
    train_gene_mean = np.mean(train_x_log, axis=0, keepdims=True)
    train_gene_std = np.std(train_x_log, axis=0, keepdims=True)    
    
    train_x = (train_x_log - train_gene_mean)/train_gene_std
    val_x = (val_x_log - train_gene_mean)/train_gene_std
    test_x = (test_x_log - train_gene_mean)/train_gene_std
    
    return train_x, train_y, val_x, val_y, test_x, test_y
    
    
class DenseNN(nn.Module):
    def __init__(self, seq_len, num_classes, drop_rate, ndim = 256):
        super(DenseNN, self).__init__()
        self.drop_rate = drop_rate
        
        layer1 = []
        layer1.append(nn.Linear(seq_len, ndim))
        layer1.append(nn.BatchNorm1d(ndim))
        layer1.append(nn.Dropout(self.drop_rate))
        layer1.append(nn.ReLU(inplace=True))
        
        layer2 = []
        layer2.append(nn.Linear(ndim, int(ndim/2)))
        layer2.append(nn.BatchNorm1d(int(ndim/2)))
        layer2.append(nn.Dropout(self.drop_rate))
        layer2.append(nn.ReLU(inplace=True))
        
        self.last = nn.Linear(int(ndim/2), num_classes)
        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        
    def forward(self, x):
        
        out = self.l1(x)
        out = self.l2(out)
        out = self.last(out)
        
        return out
    
    
def compute_acc_surv(model, inputs, ylabel, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(inputs).float().to(device)).detach().cpu().numpy()
    accuracy = np.mean(np.argmax(pred, axis=1) == ylabel)
    return accuracy


def fit_one_step(model, loss_fn, optimizer, train_X, train_Y, eta, 
                 batch_size, device):
    
    n, p = train_X.shape
    batch_xs = train_X + np.random.randn(n, p) * eta
    batch_ys = train_Y
    batch_ys = np.array([np.argmax(a) for a in batch_ys])

    torch_data = torch.utils.data.TensorDataset(torch.tensor(batch_xs).float(), torch.tensor(batch_ys))
    loader = torch.utils.data.DataLoader(dataset=torch_data, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train(True)

    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(loader):
        # Every data instance is an input + label pair
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        total += inputs.size(0)
        _, pred = torch.max(outputs.data, axis=1)
        correct += (pred == labels).sum().item()
    
    return model


def init_train(train_X, train_Y, eta, num_classes, lr, epochs, 
               batch_size, device, dropout=0.5):

    n_cell,in_dim = train_X.shape

    model = DenseNN(in_dim, num_classes, drop_rate=dropout).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    e = 1
    model = fit_one_step(model, loss_fn, optimizer, train_X, train_Y, eta, 
                         batch_size, device)
    while e < epochs:    
        e += 1
        model = fit_one_step(model, loss_fn, optimizer, train_X, train_Y, eta, 
                             batch_size, device)
        
    w1 = model.l1[0].weight.data
    w2 = model.l2[0].weight.data
    w_last = model.last.weight.data
    b1 = model.l1[0].bias.data
    b2 = model.l2[0].bias.data
    b_last = model.last.bias.data
    batch_w1 = model.l1[1].weight.data
    batch_w2 = model.l2[1].weight.data
    batch_b1 = model.l1[1].bias.data
    batch_b2 = model.l2[1].bias.data
    
    weight_bias = (w1, w2, w_last, b1, b2, b_last, batch_w1, batch_w2, batch_b1, batch_b2)
    return model, weight_bias


def variable_selet_one_step(train_X, train_Y,  val_X, val_Y,  num_classes, lr, epochs,
                           batch_size, remain_var, w1, w2, w_last, b1, b2, b_last, 
                           batch_w1, batch_w2, batch_b1, batch_b2, p, q, p0, q0, p00,
                           device, eta, dropout=0.5, elimination_rate=0.1, cut_off=0.05):

    n_cell, in_dim = train_X.shape

    model = DenseNN(in_dim, num_classes, drop_rate=dropout).to(device)
    
    model.l1[0].weight.data = w1
    model.l2[0].weight.data = w2
    model.last.weight.data = w_last
    model.l1[0].bias.data = b1
    model.l2[0].bias.data = b2
    model.last.bias.data = b_last
    model.l1[1].weight.data = batch_w1
    model.l2[1].weight.data = batch_w2
    model.l1[1].bias.data = batch_b1
    model.l2[1].bias.data  = batch_b2
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    e = 1
    model = fit_one_step(model, loss_fn, optimizer, train_X, train_Y, eta,
                         batch_size, device)

    while e < epochs:   
        e += 1
        model = fit_one_step(model, loss_fn, optimizer, train_X, train_Y, eta,
                             batch_size, device)
       
    val_YY = np.array([np.argmax(a) for a in val_Y])
    val_YY = Variable(torch.tensor(val_YY).to(device))
    val_XX = Variable(torch.tensor(val_X).float().to(device), requires_grad=True)

    model.eval()

    out = model(val_XX)
    out = loss_fn(out, val_YY)
    grad = torch.autograd.grad(outputs=out,
                               inputs=val_XX,
                               grad_outputs=torch.ones(out.size()).to(device))[0]
    s = torch.mean(torch.abs(grad), axis=0)
    num = in_dim - math.ceil(elimination_rate * (q-cut_off*p*q0/p0))
    
    idx = torch.argsort(s)[-num:].detach().cpu().numpy()
    train_X = train_X[:,idx]
    val_X = val_X[:,idx]
    remain_var = remain_var[idx]

    q = sum(remain_var > p00 - 1)
    p = num - q

    eFDR = q/p*p0/q0

    w1 = model.l1[0].weight.data[:, idx]
    w2 = model.l2[0].weight.data
    w_last = model.last.weight.data
    b1 = model.l1[0].bias.data
    b2 = model.l2[0].bias.data
    b_last = model.last.bias.data
    batch_w1 = model.l1[1].weight.data
    batch_w2 = model.l2[1].weight.data
    batch_b1 = model.l1[1].bias.data
    batch_b2 = model.l2[1].bias.data
    weight_bias = (w1, w2, w_last, b1, b2, b_last, batch_w1, batch_w2, batch_b1, batch_b2)
    
    return train_X,  val_X, eFDR, remain_var, p, q, weight_bias


def scFSNN(train_X, train_Y,  val_X, val_Y, test_X, test_Y, 
         num_classes, lr, epochs, batch_size, q0, device,
         dropout=0.5, eta=1, elimination_rate=1, cut_off=0.1):

    eFDR = 1
    EFDR=[1]

    q = q0
    p = train_X.shape[1]
    Q = [q]
    P = [p]
    p00 = p
    
    tv_X = train_X
    pool_X = tv_X.flatten()
    rand = np.random.choice(len(pool_X),(tv_X.shape[0],q))
    new_X = pool_X[rand]
    train_X = np.column_stack((train_X,new_X))
    
    tv_X = val_X
    pool_X = tv_X.flatten()
    rand = np.random.choice(len(pool_X),(tv_X.shape[0],q))
    new_X = pool_X[rand]
    val_X = np.column_stack((val_X,new_X))
 
    in_dim = train_X.shape[1]
    remain_var = np.arange(in_dim)

    model, weight_bias = init_train(train_X, train_Y, eta, 
                               num_classes, lr, 30,
                               batch_size, device, 
                               dropout=dropout)

    val_YY = np.array([np.argmax(a) for a in val_Y])
    val_YY = Variable(torch.tensor(val_YY).to(device))
    val_XX = Variable(torch.tensor(val_X).float().to(device), requires_grad=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    model.eval()

    out = model(val_XX)
    out = loss_fn(out, val_YY)
    grad = torch.autograd.grad(outputs=out,
                               inputs=val_XX,
                               grad_outputs=torch.ones(out.size()).to(device))[0]
    s = torch.mean(torch.abs(grad), axis=0)
    s = s.detach().cpu().numpy()
    s1 = s[:p]
    s2 = s[-q:]
    p0 = np.min([np.sum(np.median(s2) > s1) * 2, p])
    print(p0)

    i = 0
    while eFDR > cut_off:
        w1 = weight_bias[0] 
        w2 = weight_bias[1] 
        w_last = weight_bias[2]
        b1 = weight_bias[3] 
        b2 = weight_bias[4] 
        b_last = weight_bias[5] 
        batch_w1 = weight_bias[6] 
        batch_w2 = weight_bias[7]
        batch_b1 = weight_bias[8]
        batch_b2 = weight_bias[9]
                           
        train_X, val_X, eFDR, remain_var, p, q, weight_bias = \
            variable_selet_one_step(train_X, train_Y, val_X, val_Y, num_classes, 
                                    lr, epochs, batch_size, remain_var, w1, w2, 
                                    w_last, b1, b2, b_last, batch_w1, batch_w2,
                                    batch_b1, batch_b2, p, q, p0, q0, p00, device,
                                    eta, dropout=dropout, elimination_rate=elimination_rate, 
                                    cut_off=cut_off)
        P.append(p)
        Q.append(q)
        EFDR.append(eFDR)
        
        if i%10 == 0:
            print("eFDR:", eFDR, "p:", p, "q:", q)
        i += 1
        
    idx_org = remain_var <= p00-1
    select_var = remain_var[idx_org]

    train_X = train_X[:, idx_org]
    test_X = test_X[:, select_var]
    
    n_cell, in_dim = train_X.shape

    model = DenseNN(in_dim, num_classes, drop_rate=dropout).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.l1[0].weight.data = weight_bias[0][:, idx_org] 
    model.l2[0].weight.data = weight_bias[1] 
    model.last.weight.data = weight_bias[2] 
    model.l1[0].bias.data = weight_bias[3] 
    model.l2[0].bias.data = weight_bias[4] 
    model.last.bias.data = weight_bias[5] 
    model.l1[1].weight.data = weight_bias[6] 
    model.l2[1].weight.data = weight_bias[7] 
    model.l1[1].bias.data = weight_bias[8] 
    model.l2[1].bias.data  = weight_bias[9] 
    
    test_YY = np.array([np.argmax(a) for a in test_Y]) 
    acc_test_trace = []

    e = 1
    while e < 10:
        e += 1
        model = fit_one_step(model, loss_fn, optimizer, train_X, train_Y, eta,
                                                batch_size, device)
        acc_test_trace.append(compute_acc_surv(model, test_X, test_YY, device))
    
    acc_test = compute_acc_surv(model, test_X, test_YY, device)
    
    print("test accuracy:", acc_test)
    
    return select_var, acc_test, acc_test_trace, P, Q, EFDR


