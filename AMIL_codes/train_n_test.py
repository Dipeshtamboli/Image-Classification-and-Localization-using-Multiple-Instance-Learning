from __future__ import print_function

import numpy as np
from scipy.misc import imsave
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

# from dataloader import MnistBags
from amil_model import Attention

# from __future__ import print_function, division
import os
import glob
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from amil_model import Attention
from patch_data import PatchMethod
from tensorboardX import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='Breakthis data_mynet')
parser.add_argument("--zoom", help='zoom_level',default=400)
parser.add_argument('--epochs',type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

zoom_level_x =str(args.zoom) + 'X'


print('zoom_level_{} epoch_{} learning_rate_{}'.format(zoom_level_x, args.epochs, args.lr))
writer = SummaryWriter(zoom_level_x+'/runs/'+"epoch:"+str(args.epochs))

# Training settings
# parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
# parser.add_argument('--epochs', type=int, default=1, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
#                     help='learning rate (default: 0.01)')
# parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
#                     help='weight decay')
# parser.add_argument('--target_number', type=int, default=9, metavar='T',
#                     help='bags have a positive labels if they contain at least one 9')
# parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
#                     help='average bag length')
# parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
#                     help='variance of bag length')
# parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
#                     help='number of bags in training set')
# parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
#                     help='number of bags in test set')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
#                                                mean_bag_length=args.mean_bag_length,
#                                                var_bag_length=args.var_bag_length,
#                                                num_bag=args.num_bags_train,
#                                                seed=args.seed,
#                                                train=True),
#                                      batch_size=1,
#                                      shuffle=True,
#                                      **loader_kwargs)

# test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
#                                               mean_bag_length=args.mean_bag_length,
#                                               var_bag_length=args.var_bag_length,
#                                               num_bag=args.num_bags_test,
#                                               seed=args.seed,
#                                               train=False),
#                                     batch_size=1,
#                                     shuffle=False,
#                                     **loader_kwargs)

# dir structure would be: data/class_name(0 and 1)/dir_containing_img/img
# sftp://test1@10.107.42.42/home/Drive2/amil/data
# /home/dipesh/test1/new_data_tree/40X
# /home/dipesh/126/AMIL_project/AMIL_codes/amil_model.py
# /home/dipesh/126/AMIL_project/
data_path_train = "../AMIL_Data/{}/train".format(zoom_level_x)
data_path_test = "../AMIL_Data/{}/test".format(zoom_level_x)

data = PatchMethod(root = data_path_train)
val_data =PatchMethod(root = data_path_test, mode = 'test')
# data = PatchMethod(root = '/Users/abhijeetpatil/Desktop/screenshots2/')
# val_data =PatchMethod(root = '/Users/abhijeetpatil/Desktop/screenshots2/', mode = 'test')

train_loader = torch.utils.data.DataLoader(data, shuffle = True, num_workers = 6, batch_size = 1)
test_loader = torch.utils.data.DataLoader(val_data, shuffle = False, num_workers = 6, batch_size = 1)


print('Init Model')
model = Attention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
# optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    correct_label_pred = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # print("label",label[0][0])
        # print("label",label[0])
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        data = data.squeeze(0)

        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data[0]
        error, predicted_label_train = model.calculate_classification_error(data, bag_label)
        # print("bag_label,predicted_label_train",bag_label,predicted_label_train)
        # print(int(bag_label) == int(predicted_label_train))
        correct_label_pred += (int(bag_label) == int(predicted_label_train))
        # exit()
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        # print(correct_label_pred)
        # print(len(train_loader))
        # print(batch_idx)

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    train_acc = (1 - train_error)*100

    writer.add_scalar('data/train_acc', train_acc, epoch)
    writer.add_scalar('data/train_error', train_error, epoch)
    writer.add_scalar('data/train_loss', train_loss, epoch)

    result_train = 'Epoch: {}, Loss: {:.4f}, Train error: {:.4f}, Train accuracy: {:.2f}'.format(epoch, train_loss.cpu().numpy()[0], train_error, train_acc)

    print(result_train)
    return result_train

def test(epoch):
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        # print(label)
        # print((data[0].shape))

        bag_label = label[0]
        instance_labels = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

        visualization_attention(data[0],attention_weights[0],batch_idx,epoch)
        if batch_idx < 1:  # plot bag labels and instance labels for first 5 bags
            bag_level = (bag_label.cpu().data.numpy(), int(predicted_label.cpu().data.numpy()))
            # print(bag_level)
            # print(instance_labels)
            # visualization_attention(data[0],attention_weights[0],batch_idx,epoch)
            # print("attention_weights.shape",attention_weights.shape)
            # instance_level = list(zip(instance_labels.numpy().tolist(),
            #                      np.round(attention_weights.cpu().data.numpy(), decimals=3).tolist()))

            # print('\nTrue Bag Label, Predicted Bag Label: {}\n'
            #       'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    test_acc = (1 - test_error)*100    

    writer.add_scalar('data/test_acc', test_acc, epoch)
    writer.add_scalar('data/test_error', test_error, epoch)
    writer.add_scalar('data/test_loss', test_loss, epoch)
    result_test = 'Epoch: {}, Loss: {:.4f}, test error: {:.4f}, test accuracy: {:.2f}'.format(epoch, test_loss.cpu().numpy()[0], test_error, test_acc)
    print(result_test)
    return result_test
    # print('Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))

def visualization_attention(data,attention_weights,batch_idx,epoch):
    img_save_dir = './{}/AMIL_visualization/epoch_{}'.format(zoom_level_x,epoch)
    img_save_name = img_save_dir + '/test_epoch_{}_no_{}.png'.format(epoch,batch_idx)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    data = data.cpu().data.numpy()
    attention_weights = attention_weights.cpu().data.numpy()
    # print("data.shape",data.shape)
    # print("attention_weights",attention_weights.shape)
    attention_weights = attention_weights/np.max(attention_weights)
    complete_image=np.zeros((3,480,700))
    for height_no in range(16):
        for width_no in range(25):
            complete_image[:,height_no*28:height_no*28+28, width_no*28:width_no*28+28] = data[height_no*25+width_no,:,:,:] * attention_weights[height_no*25+width_no]
    complete_image = complete_image.transpose((1,2,0))
    imsave(img_save_name,complete_image)
    # weighted_images_list = data * attention_weights



if __name__ == "__main__":
    # img_save_dir = './AMIL_visualization/zoom_{}/epoch_{}'.format(zoom_level_x,epoch)
    main_dir = "./" + zoom_level_x +'/'
    folders = ["pt_files","txt_file"]
    for i in folders:
        if not os.path.exists(main_dir + i ):
            os.makedirs(main_dir + i )

    save_string="AMIL_Breakthis_epochs: "+str(args.epochs)+"zoom:"+zoom_level_x
    save_name_txt = main_dir+"txt_file/"+save_string+".txt"

    model_file = open(save_name_txt,"w") 
    for epoch in range(1, args.epochs + 1):
        print('----------Start Training----------')
        train_result = train(epoch)
        print('----------Start Testing----------')
        test_result = test(epoch)
        model_file.write(test_result + '\n')
        model_file.write(train_result + '\n')
    model_file.close()
    torch.save(model.state_dict(),main_dir+"pt_files/"+save_string+"AMIL_Breakthis_state_dict.pt")
    torch.save(model ,main_dir+"pt_files/"+save_string+"AMIL_Breakthis_model.pt")
