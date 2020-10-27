import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from SV_utils import powerset #tool box fo FL_SV
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from itertools import permutations 
import sys
import csv

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            print('data is iid')
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
            print('data is non-iid')

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

    else:
        exit('Error: unrecognized model')
    print(net_glob)

    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

#####################################################################################################
    #manipulate the order of choosing clients
    client_lst = list(range(0, args.num_users))
    print(client_lst)
    idx_to_clients = {}        #idx to clients
    subset_model_dict = {}     #clients to models
    all_subset = powerset(client_lst)  #record all subset of a client list
    all_subset.remove([])
    # print(len(all_subset))

    for i in range(len(all_subset)):
        idx_to_clients[i] = all_subset[i]


    for i in range(len(all_subset)):
        stamp = tuple(idx_to_clients[i])
        subset_model_dict[stamp] = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        subset_model_dict[stamp].train()

    print('all subset model has been initialized')
    # print(idx_to_clients,subset_model_dict)
    # sys.exit(0)
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        loss_locals = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = client_lst

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        #Updateing recording model
        for i in range(len(all_subset)):
            curr_subset = idx_to_clients[i]  #get the clients for current users
            curr_model = subset_model_dict[tuple(curr_subset)]  #get the current model
            # print(curr_subset)
            # print(curr_model)
            u_w_locals = []
            u_loss_locals = []
            # calculate the local update for recording model
            for user in curr_subset:
                u_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user])
                u_w, u_loss = u_local.train(net=copy.deepcopy(subset_model_dict[tuple(curr_subset)]).to(args.device))
                u_w_locals.append(copy.deepcopy(u_w))
                u_loss_locals.append(copy.deepcopy(u_loss))

            # update delta_w for recording model
            u_w_glob = FedAvg(u_w_locals)

            # copy weight to recording model
            subset_model_dict[tuple(curr_subset)].load_state_dict(u_w_glob)
        print('local model updates has been done')


    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    #get the testing acc for each subset
    acc_record_dict = {}
    for i in range(len(all_subset)):
        curr_subset = idx_to_clients[i]  #get the clients for current users
        # curr_model = subset_model_dict[tuple(curr_subset)]  #get the current model 
        subset_model_dict[tuple(curr_subset)].eval()
        u_acc_test, u_loss_test = test_img(subset_model_dict[tuple(curr_subset)], dataset_test, args)
        acc_record_dict[tuple(curr_subset)] = u_acc_test

    print(len(acc_record_dict),acc_record_dict)

    #save the dict into csv file
    with open('exact_dict.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in acc_record_dict.items():
           writer.writerow([key, value])
    #initialize a mc dict contains all clients
    mc_dict = {}
    for i in range(len(client_lst)):
        mc_dict[i] = 0
    #calculate the marginal contribution of each client
    






