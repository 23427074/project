from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F
import sys
import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data.sampler import Sampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
# from Data_Loader import Labels_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from matplotlib.pyplot import MultipleLocator 
import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, weights_init
from losses import bce_dice_loss, dice_loss, focal_loss, tversky_loss, focal_tversky_loss 
from losses import threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow, show_training_dataset
from Metrics import dice_coeff, accuracy_score
import time
from numba import cuda
import Unet3
#from ploting import VisdomLinePlotter
#from visdom import Visdom


#######################################################
#Checking if GPU is used
#######################################################'

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#tch -n 1 nvidia-smi######################################################
#Setting the basic paramters of the model
#######################################################

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print('GPU count: ', torch.cuda.device_count())

batch_size = 8
print('batch_size = ' + str(batch_size))

valid_size = 0.15
# print('valid percent = ', valid_size)

epoch = 30
print('epoch = ' + str(epoch))
print('valid percent = ', valid_size)
random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 4
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True

#plotter = VisdomLinePlotter(env_name='Tutorial Plots')

#######################################################
#Setting up the model
#######################################################

model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]


def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

#passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary


#model_test = model_unet(model_Inputs[4], 3, 1)
#model_test.apply(weights_init)
model_test = model_unet(NestedUNet, 3, 1)
model_test.apply(weights_init)
#model_test = model_unet(Unet3.UNet_3Plus, 3, 1)

#model_test = torch.nn.DataParallel(model_test)

model_test.to(device)

#######################################################
#Getting the Summary of Model
#######################################################

torchsummary.summary(model_test, input_size=(3, 128, 128))

#######################################################
#Passing the Dataset of Images and Labels
#######################################################

t_data = '/home/kevin295643815697236/nested_unet/Unet/data/ISLE/image24'
l_data = '/home/kevin295643815697236/nested_unet/Unet/data/ISLE/mask5'
#test_image = '/home/kevin295643815697236/nested_unet/Unet/data/TGVH/new_image/00816.png'
#test_label = '/home/kevin295643815697236/nested_unet/Unet/data/TGVH/new_mask/00816.png'
test_folderP = '/home/kevin295643815697236/nested_unet/Unet/data/ISLE/new_image20/*'
test_folderL = '/home/kevin295643815697236/nested_unet/Unet/data/ISLE/new_mask/*'


data_transform = torchvision.transforms.Compose([
                        #torchvision.transforms.Resize((128,128)),
                        torchvision.transforms.CenterCrop(96),
                        torchvision.transforms.RandomRotation((-10,10)),
                        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        # torchvision.transforms.Grayscale(),
                        # torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                    ])



# print(t_data + '/*')
Training_Data = Images_Dataset_folder(t_data,
                                      l_data, # 
                                      )

# Mask_Data = Labels_Dataset_folder(l_data,) # 

# print(Training_Data[1])

'''
read_test_folder = '.see'


if os.path.exists(read_test_folder) and os.path.isdir(read_test_folder):
    shutil.rmtree(read_test_folder)

try:
    os.mkdir(read_test_folder)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder)
else:
    print("Successfully created the testing directory %s " % read_test_folder)
    '''

'''
for i in range(len(Training_Data)):
    seedata = plt.imsave(
                './see' + str(i) + '.png', Training_Data[i])'''
# show_training_dataset(Training_Data)


# Training_Data.__len__()
# Training_Data.__getitem__(0)
#  print(Training_Data)

#######################################################
#Giving a transformation for input data
#######################################################

data_transform1 = torchvision.transforms.Compose([
          #  torchvision.transforms.Resize((128,128)),
         #   torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

data_transform2 = torchvision.transforms.Compose([
              #  torchvision.transforms.Resize((128,128)),
                      #    torchvision.transforms.CenterCrop(96),
                 torchvision.transforms.Grayscale(),
                                   #            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                           ])


#######################################################
#Trainging Validation Split
#######################################################
num_train = len(Training_Data)
print('Total Data =', num_train)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
# print('Num of Valid Data: ', split)

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]

print('Num of Training Data = ', len(train_idx))
print('Num of Valid Data =: ', len(valid_idx))

'''
tv_dir = os.listdir(t_data)

train, valid = tv_dir[split:], tv_dir[:split]
#print(train)
#print(valid)
for t in train:
    train_no = []
    #print(train_dir + '/' + x)
    #tra = t_data + '/' + t
    train_no.append(t)
    # print(train_full_dir)

for v in valid:
    valid_no = []
    # va = t_data + '/' + v
    valid_no.append(v)'''
   
# print(train_full_dir)



train_sampler = RandomSampler(train_idx)
#print(type(train_sampler))
#print(train_sampler)
#train_mask_sampler = Se`1quentialSampler(train_mask_idx) #
# print('train_sampler', len(train_sampler))
valid_sampler = RandomSampler(valid_idx)
#valid_mask_sampler = SequentialSampler(valid_mask_idx) #
# print('valid_sampler', len(valid_sampler))


train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=8, sampler=train_sampler,
                                           num_workers=8, pin_memory=pin_memory,)
                        

#print(type(train_loader))

#train_mask_loader = torch.utils.data.DataLoader(Mask_Data, batch_size=4, sampler=train_sampler,
                                            #num_workers=num_workers, pin_memory=pin_memory,) #

# show_training_dataset(train_loader)
#print('train data num for each batch =', len(train_loader))

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=8, sampler=valid_sampler,
                                           num_workers=8, pin_memory=pin_memory,)

#valid_mask_loader = torch.utils.data.DataLoader(Mask_Data, batch_size=2, sampler=valid_sampler,
                                                   # num_workers=num_workers, pin_memory=pin_memory,) #

#print('valid data num for each batch =', len(valid_loader))

'''
print('train')
print(train_loader)
print('valid')
print(valid_loader)
'''

#######################################################
#Using Adam as Optimizer
#######################################################

initial_lr = 0.001
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr, weight_decay=0.0) # try SGD
#opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.9)


scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = 10, gamma=0.1)
#scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)

#######################################################
#Writing the params to tensorboard
#######################################################

#writer1 = SummaryWriter()
#dummy_inp = torch.randn(1, 3, 128, 128)
#model_test.to('cpu')
#writer1.add_graph(model_test, model_test(torch.randn(3, 3, 128, 128, requires_grad=True)))
#model_test.to(device)

#######################################################
#Creating a Folder for every data of the program
#######################################################

New_folder = './model'

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

#######################################################
#Setting the folder of saving the predictions
#######################################################

read_pred = './model/pred'

#######################################################
#Checking if prediction folder exixts
#######################################################

if os.path.exists(read_pred) and os.path.isdir(read_pred):
    shutil.rmtree(read_pred)

try:
    os.mkdir(read_pred)
except OSError:
    print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
else:
    print("Successfully created the prediction directory '%s' of dice loss" % read_pred)

#######################################################
#checking if the model exists and if true then delete
#######################################################

read_model_path = './model/Unet_D_' + str(epoch) + '_' + str(batch_size)

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)


read_test_folder_test = './model/test'


if os.path.exists(read_test_folder_test) and os.path.isdir(read_test_folder_test):
        shutil.rmtree(read_test_folder_test)

try:
    os.mkdir(read_test_folder_test)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_test)
else:
    print("Successfully created the testing directory %s " % read_test_folder_test)

#######################################################
#Training loop
#######################################################

total_dice = []

#train_idx = os.listdir(t_data)


#lossf = tversky_loss()
#lossf = bce_dice_loss()
#lossf = focal_loss()
lossf = focal_tversky_loss()

for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()
    # scheduler.step(i)
    # lr = scheduler.get_last_lr()

    #######################################################
    #Training Data
    #######################################################
    # print('0')
    model_test.train()
    k = 1
    # run = 0
    # print('Start Training...')
    # 假設我有24個data要train
    # batch size 是 4
    # 則train_loader的size是6
    # for loop跑6次每次去train 4個data x是image y是mask

    # train_len = os.listdir(t_data)


    #for x in train_loader:
        #print(type(x))


    #for tra,va in zip(train_no, train_no):
    #for x,y in zip(train_loader, train_mask_loader):
        #print(x[0])
    #for steps, train in enumerate(train_loader): #
    for x, y in train_loader:
        
        x, y = x.to(device), y.to(device)
        #If want to get the input images with their Augmentation - To check the data flowing in net
        input_images(x, y, i, n_iter, k)

        # print('3')
        '''
        grid_img = torchvision.utils.make_grid(x)
        writer1.add_image('images', grid_img, 0)

        grid_lab = torchvision.utils.make_grid(y)'''
        opt.zero_grad() 
        y_pred = model_test(x)
        #print(type(y_pred))
        lossT = lossf(y_pred, y)# Dice_loss Used
        # print("lossT:", lossT)
        # print("lossT.item(): ",lossT.item())
        #print(type(lossT))
        #lossT = torch.from_numpy(lossT)
        #print(type(lossT))
        train_loss += lossT.item() * x.size(0)
        # print("train_loss: ",train_loss)
        # print("x_size(0): ", x_size(0))
        lossT.backward()
      #  plot_grad_flow(model_test.named_parameters(), n_iter)
        opt.step()
        scheduler.step()
        lr = scheduler.get_last_lr()
        x_size = lossT.item() * x.size(0)
        # print("x_size: ",x_size)
        k = 2
       #  run+=1
   #  print('times: ', run)

    #    for name, param in model_test.named_parameters():
    #        name = name.replace('.', '/')
    #        writer1.add_histogram(name, param.data.cpu().numpy(), i + 1)
    #        writer1.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), i + 1)


    #######################################################
    #Validation Step
    #######################################################

    model_test.eval()
    torch.no_grad() #to increase the validation process uses less memory
    # print('Validing Data...')


    #for step, valid in enumerate(valid_loader): #
    for x1, y1 in valid_loader:
        #print('2')
       #  x1, y1 = x1.to(device), y1.to(device)
        #print(step)
        #x1, y1 = valid
        x1, y1 = x1.to(device), y1.to(device)
        #print('3')
        y_pred1 = model_test(x1)
        #type(y_pred1)
        lossL = lossf(y_pred1, y1)# Dice_loss Use
        # print("lossL: ", lossL)
        # print("lossL.item(): ", lossL.item())
        valid_loss += lossL.item() * x1.size(0)
        # print("valid_loss: ", valid_loss)
        x_size1 = lossL.item() * x1.size(0)
        # print("x_size1: ", x_size1)

    #######################################################
    #Saving the predictions
    #######################################################

  #  accuracy = accuracy_score(pred_tb[0][0], s_label)

    #######################################################
    #To write in Tensorboard
    #######################################################
    # print("total_train_loss: ", train_loss)
    train_loss = train_loss / len(train_idx)
    valid_loss = valid_loss / len(valid_idx)

    if (i+1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))
 #       writer1.add_scalar('Train Loss', train_loss, n_iter)
  #      writer1.add_scalar('Validation Loss', valid_loss, n_iter)
        #writer1.add_image('Pred', pred_tb[0]) #try to get output of shape 3


    #######################################################
    #Early Stopping
    #######################################################
    # print("epoch_valid: ",epoch_valid)
    # print("i: ", i)
    if valid_loss <= valid_loss_min and epoch_valid >= i: # and i_valid <= 2:
        # print("epoch_valid: ", epoch_valid)
        # print("i: ", i)
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(),'./model/Unet_D_' +
                                              str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
                                              + '_batchsize_' + str(batch_size) + '.pth')
       # print(accuracy)
        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print("i_valid: ",i_valid)
            i_valid = i_valid+1
        valid_loss_min = valid_loss
        #if i_valid ==3:
         #   break

    #######################################################
    # Extracting the intermediate layers
    #######################################################

    #####################################
    # for kernals
    #####################################
    x1 = torch.nn.ModuleList(model_test.children())
    # x2 = torch.nn.ModuleList(x1[16].children())
     #x3 = torch.nn.ModuleList(x2[0].children())

    #To get filters in the layers
   # plot_kernels(x1.weight.detach().cpu(), 7)

    #####################################
    # for images
    #####################################
    '''
    x2 = len(x1)
    dr = LayerActivations(x1[x2-1]) #Getting the last Conv Layer

    img = Image.open(test_image)
    s_tb = data_transform(img)

    pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()

    #plot_kernels(dr.features, n_iter, 7, cmap="rainbow")'''

    #####################################
    # 每個epoch就紀錄一次dice score
    #####################################

    dice_score12 = 0.0
    img_test_no = 0

    read_test_folder = glob.glob(test_folderP)
    x_sort_test = natsort.natsorted(read_test_folder)  # To sort

    read_test_folderL = glob.glob(test_folderL)
    x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort

    # pred = [None]*5
   #  print(len(pred))

    for m in range(len(read_test_folder)):
            # print('m: ', m)
            im = Image.open(x_sort_test[m])
            s = data_transform1(im)
            pred = model_test(s.unsqueeze(0).cuda()).cpu()
            pred = torch.sigmoid(pred)
            pred = pred.detach().numpy()

            x1 = plt.imsave('./model/test/im_epoch_' + str(epoch) + 'int_' + str(m)
                                        + '_img_no_' + '.png', pred[0][0], cmap = 'gray')
    # print(len(pred))
    # print(pred[3])
    read_test_folderK = glob.glob('./model/test/*')
    x_sort_testK = natsort.natsorted(read_test_folderK)
    # print(x_sort_testK)

    for n in range(len(read_test_folderK)):
           #  print('n: ', n)
            x = Image.open(x_sort_testK[n])
            # s = Image.fromarray(pred[0][0])
            # s = pred.squeeze()
            s = data_transform2(x)
            s = np.array(s)
            s = threshold_predictions_v(s)

            y = Image.open(x_sort_testL[n])
            s2 = data_transform2(y)
            s3 = np.array(s2)

            total = dice_coeff(s, s3)
            # print(total)

            dice_score12 = dice_score12 + total

    dice_epoch = dice_score12/len(read_test_folder)

    print(' Average Dice Score : ' + str(dice_epoch))
    total_dice.append(dice_epoch)
    # print(total_dice)

    
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    n_iter += 1

plt.figure(0)
plt.plot(range(1,epoch+1,1), np.array(total_dice), 'r-', label= "dice_score") #relative global step
plt.xlabel('epoch')
plt.ylabel('dice')
y_major = MultipleLocator(0.1)
x_major = MultipleLocator(10)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major)
ax.yaxis.set_major_locator(y_major)
plt.ylim(0.05,1.0)
plt.xlim(-0.05,102)
# plt.legend()
plt.savefig(f"./model/dice_score.png")

#######################################################
#closing the tensorboard writer
#######################################################

#writer1.close()

#######################################################
#if using dict
#######################################################

#model_test.filter_dict

#######################################################
#Loading the model
#######################################################

test1 =model_test.load_state_dict(torch.load('./model/Unet_D_' +
                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))


#######################################################
#checking if cuda is available
#######################################################

if torch.cuda.is_available():
    torch.cuda.empty_cache()

#######################################################
#Loading the model
#######################################################

model_test.load_state_dict(torch.load('./model/Unet_D_' +
                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))

model_test.eval()

#######################################################
#opening the test folder and creating a folder for generated images
#######################################################

read_test_folder = glob.glob(test_folderP)
x_sort_test = natsort.natsorted(read_test_folder)  # To sort


read_test_folder112 = './model/gen_images'


if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
    shutil.rmtree(read_test_folder112)

try:
    os.mkdir(read_test_folder112)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder112)
else:
    print("Successfully created the testing directory %s " % read_test_folder112)


#For Prediction Threshold

read_test_folder_P_Thres = './model/pred_threshold'


if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

#For Label Threshold

read_test_folder_L_Thres = './model/label_threshold'


if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)

try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)


'''
read_test_folder_T_Thres = './model/test_pics'


if os.path.exists(read_test_folder_T_Thres) and os.path.isdir(read_test_folder_T_Thres):
    shutil.rmtree(read_test_folder_T_Thres)

try:
    os.mkdir(read_test_folder_T_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_T_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_T_Thres)
'''
'''
for in range(len(train_loader)):
             seedata = plt.imsave(
                        './see' + str(i) + '.png', train_loader[i])'''

#######################################################
#saving the images in the files
#######################################################

print('Counting Dice Loss for Testing Data...')

img_test_no = 0

for i in range(len(read_test_folder)):
    im = Image.open(x_sort_test[i])

    im1 = im
    im_n = np.array(im1)
    im_n_flat = im_n.reshape(-1, 1)

    for j in range(im_n_flat.shape[0]):
        if im_n_flat[j] != 0:
            im_n_flat[j] = 255

    s = data_transform1(im)
    pred = model_test(s.unsqueeze(0).cuda()).cpu()
    pred = torch.sigmoid(pred)
    pred = pred.detach().numpy()

#    pred = threshold_predictions_p(pred) #Value kept 0.01 as max is 1 and noise is very small.

    if i % 24 == 0:
        img_test_no = img_test_no + 1

    x1 = plt.imsave('./model/gen_images/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', pred[0][0], cmap = 'gray')


####################################################
#Calculating the Dice Score
####################################################

read_test_folderP = glob.glob('./model/gen_images/*')
x_sort_testP = natsort.natsorted(read_test_folderP)
# x_sort_testP[x_sort_testP > 0] = 1


read_test_folderL = glob.glob(test_folderL)
x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort
# print(np.max(x_sort_testL))
# tmp = np.array(x_sort_testL)
# tmp[tmp > 0] = 1
# x_sort_testL = Image.fromarray(tmp)


dice_score123 = 0.0
x_count = 0
x_dice = 0
# print('Counting Dice Loss for Testing Data...')

for i in range(len(read_test_folderP)):
    #print(len(read_test_folderP))
    x = Image.open(x_sort_testP[i])
    # x = x.convert('L')
   #  print(x)
    s = data_transform2(x)
    s = np.array(s)
    s = threshold_predictions_v(s)
    # np.set_printoptions(threshold=sys.maxsize)
    # print(s)
    

    #save the images
    x1 = plt.imsave('./model/pred_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s, cmap = "gray")

    y = Image.open(x_sort_testL[i])
    s2 = data_transform2(y)
    s3 = np.array(s2)
   # s2 =threshold_predictions_v(s2)

    #save the Images
    y1 = plt.imsave('./model/label_threshold/im_epoch_' +str(epoch) + 'int_' + str(i)
            + '_img_no_' + str(img_test_no) + '.png', s3, cmap = 'gray')

    total = dice_coeff(s, s3)
    print('Testing Data ' + str(i) + ' Dice = '+ str(total))

    if total <= 0.3:
        x_count += 1
    if total > 0.3:
        x_dice = x_dice + total
    dice_score123 = dice_score123 + total


print('Averge Dice Score = ' + str(dice_score123/len(read_test_folderP)))
#print(x_count)
#print(x_dice)
