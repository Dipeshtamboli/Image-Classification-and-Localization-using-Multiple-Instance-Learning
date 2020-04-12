#model
from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time
from tensorboardX import SummaryWriter
from datetime import datetime
import pickle
import matplotlib.pyplot as plt  
# from sklearn.metrics import f1_score, classification_report, confusion_matrix

parser = argparse.ArgumentParser(description='Breakthis data_mynet')
parser.add_argument('--epochs',type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument("--zoom", help='zoom_level',default=40)

args = parser.parse_args()
zoom_level_x =str(args.zoom) + 'X'
writer = SummaryWriter(zoom_level_x+'/runs/'+"epoch:"+str(args.epochs))

transform = transforms.Compose([transforms.RandomCrop(224),transforms.ToTensor(),])

data_train = torchvision.datasets.ImageFolder("../AMIL_Data/test/{}".format(zoom_level_x), transform=transform)
print(data_train)
data_test = torchvision.datasets.ImageFolder("../AMIL_Data/train/{}".format(zoom_level_x), transform=transform)
print(data_test)

vgg_based = torchvision.models.vgg19(pretrained=True)
print(vgg_based)
for param in vgg_based.parameters():
    # print(param)
    param.requires_grad = False

# Modify the last layer
features = list(vgg_based.classifier.children())[:-1] # Remove last layer
# print(type(vgg_based.classifier.children()))
features.extend([torch.nn.Linear(4096, 2)])
vgg_based.classifier = torch.nn.Sequential(*features)



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    # print("-----------",train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(len(data),len(target))
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        # print(output.shape)
        # print(len(output), len(target))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTrain_accuracy: {:.0f}%\n'.format(100. * correct / len(train_loader.dataset)))
    writer.add_scalar('train_Accuracy_epoch',100. * correct / len(train_loader.dataset),epoch)
    writer.add_scalar('train_loss_epoch',loss/len(train_loader.dataset),epoch)
    return (100. * correct / len(train_loader.dataset))
def test(args, model, device, test_loader,epoch):
    print("test started")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_scalar('test_loss_epoch',test_loss,epoch)
    writer.add_scalar('test_Accuracy_epoch',100. * correct / len(test_loader.dataset),epoch)
    return (100. * correct / len(test_loader.dataset))

def main():
    start = time.time()
    print ("into the main")
    parser = argparse.ArgumentParser(description='Breakthis')
    
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    
    parser.add_argument('--epochs', default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    

    parser.add_argument('--lr', type=float, default=0.0003   , metavar='LR',
                        help='learning rate (default: 0.01)')
   
    parser.add_argument('--momentum', type=float, default=0.4, metavar='M',
                        help='SGD momentum (default: 0.9)')
   
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
   
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
   
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument("--zoom", help='zoom_level',default=40)

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
	#    device = "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=32 ,shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=32 ,shuffle = False, **kwargs)
    print("device: ",device)
    
    model = vgg_based.to(device)
    print ("model transferred to device")
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)   
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=args.momentum, centered=False)
    print ("optimizer choosed")
    print("#######__parameters__######")
    print("learning rate: ", args.lr, "\nmomentum: ", args.momentum, "\nepochs: ", args.epochs)
    print("############################")    
    print("model:\n",model)
    print("############################")
    print("optimizer:\n",optimizer)
    print("############################")

    # for epoch in range(2):
    for epoch in range(1, int(args.epochs) + 1):
    	# print("-----",train_loader)
    	train_acc=train(args, model, device, train_loader, optimizer, epoch)
        test_acc=test(args, model, device, test_loader, epoch)
        # writer.add_scalar('loss_fn2',loss.item(),epoch)
        # sftp://test1@10.107.42.42/home/Drive2/amil/my_network/runs

    main_dir = "./" + zoom_level_x +'/'
    folders = ["pt_files","pickel_files","txt_file"]
    for i in folders:
        if not os.path.exists(main_dir + i ):
            os.makedirs(main_dir + i )


    save_string="Breakthis: train_acc:"+str(train_acc)+" test-acc:"+str(test_acc)+" epochs: "+str(args.epochs)+"zoom:"+zoom_level_x
    if (args.save_model):
        torch.save(model.state_dict(),main_dir+"pt_files/"+save_string+"Breakthis_state_dict.pt")
        torch.save(model ,main_dir+"pt_files/"+save_string+"Breakthis_model.pt")
    save_name_pkl = main_dir+"pickel_files/"+save_string+".pkl"
    save_name_txt = main_dir+"txt_file/"+save_string+".txt"
    model_file = open(save_name_txt,"w") 
    model_string = str(model)
    optimizer_string = str(optimizer)
    model_file.write(model_string)
    model_file.write(optimizer_string)
    model_file.write(save_name_txt)
    model_file.close()
   
    f=open(save_name_pkl,"wb")
    pickle.dump(model, f)
    end = time.time()
    print('time taken is ', (end-start))


if __name__ == '__main__':
    main()
