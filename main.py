
import random
import datetime
import models_load
from base_aug import *
from data_load import *
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from utils import *
from tqdm import tqdm
import torchvision.transforms.functional as F

from utils import DisturbLabel


SEED=10
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True


NB_CLASS = 38
IMAGE_SIZE = 224
BATCH_SIZE = 32  #2250/16
DATE = str(datetime.date.today())
MODEL_NAME = 'ResNet50'


ALPHA = 1
BETA = 1


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.opt = opt
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        # with torch.cuda.stream(self.stream):
        #     for k in self.batch:
        #         if k != 'meta':
        #             self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
def getModel(path, isTrain = 1):
    print('[+] loading model... ', end='', flush=True)

    if isTrain == 0:
        model = models_load.ResNetFinetune_SpeciesLoss_weight(NB_CLASS, dropout=True)
        model.cuda()
        path_model = './model/'+MODEL_NAME+'/' + path
        model.load_state_dict(torch.load(path_model)['state_dict'])
        print('[+] loading model :'+ path_model)
        print("model loss: %f || model acc: %f ||model epoch: %d" %(torch.load(path_model)['val_loss'], torch.load(path_model)['val_correct'],
                                                                    torch.load(path_model)['epoch']))
    else:
        model = models_load.ResNetFinetune_SpeciesLoss_weight(NB_CLASS, dropout=True)
        model.cuda()
    # print('Done')
    return model

def tta_predict(input, model):  # input is a tensor(32,*,*,*)

    outputs = []
    # model
    type(input)
    print(type(input))

    outputs.append(model(F.to_tensor([F.to_pil_image(i[1]) for i in enumerate(input)])))
    outputs.append(model(F.to_tensor([F.hflip(F.to_pil_image(i[1])) for i in enumerate(input)])))
    outputs.append(model(F.to_tensor(F.vflip(input))))
    outputs.append(model(F.to_tensor(F.rotate(input, 90))))
    outputs.append(model(F.to_tensor(F.rotate(input, 180))))
    outputs.append(model(F.to_tensor(F.rotate(input, 270))))
    outputs.append(model(F.to_tensor(F.to_grayscale(input))))
    outputs.append(model(F.to_tensor(F.adjust_brightness(input, 0.4))))
    outputs.append(model(F.to_tensor(F.adjust_contrast(input, 0.4))))
    outputs.append(model(F.to_tensor(F.adjust_gamma(input, 0.4))))
    outputs.append(model(F.to_tensor(F.adjust_hue(input, 0.2))))

    out = []
    for i in enumerate(input):
        t = transforms.RandomVerticalFlip(1)
        out.append(model(t(i[1])))
    outputs.append(out)
    out = []
    for i in enumerate(input):
        t = transforms.RandomVerticalFlip(1)
        out.append(model(t(i[1])))
    outputs.append(out)

    return np.sum(outputs, axis=0)


def predict(model, dataloader, tta=False):
    all_labels_cls = []
    all_labels_sp = []
    all_outputs_cls = []
    all_outputs_sp = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            all_labels_cls.append(labels[0])
            all_labels_sp.append(labels[1])
            inputs = Variable(inputs).cuda()
            if tta:
                outputs = tta_predict(inputs, model)
            else:
                outputs_cls, outputs_sp = model(inputs)
            all_outputs_cls.append(outputs_cls.data.cpu())
            all_outputs_sp.append(outputs_sp.data.cpu())
        all_outputs_cls = torch.cat(all_outputs_cls)
        all_outputs_sp = torch.cat(all_outputs_sp)
        all_labels_cls = torch.cat(all_labels_cls)
        all_labels_sp = torch.cat(all_labels_sp)

        all_labels_cls = all_labels_cls.cuda()
        all_outputs_cls = all_outputs_cls.cuda()
        all_labels_sp = all_labels_sp.cuda()
        all_outputs_sp = all_outputs_sp.cuda()
    return all_labels_cls, all_outputs_cls, all_labels_sp, all_outputs_sp


'''
测试data在model上的结果
'''

def testTTA(model, test_list):
    test(model, test_list)

    transformsTTA = [
        transforms.RandomVerticalFlip(0),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomVerticalFlip(1),
        transforms.RandomRotation(60),
        transforms.RandomRotation(90),
        transforms.RandomRotation(120),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
    ]

    test_dataset = {}
    test_dataLoader = {}
    for i, t in enumerate(transformsTTA):
        test_dataset[i] = MyDataSet(root=test_list, transform=t, root_pre=IMAGE_PRE)
        test_dataLoader[i] = DataLoader(dataset=test_dataset[i], batch_size=BATCH_SIZE, shuffle=False)


    all_labels = [0 for i in range(len(transformsTTA))]
    all_outputs = [0 for i in range(len(transformsTTA))]
    model.eval()
    with torch.no_grad():
        for i in range(len(transformsTTA)):
            all_outputs[i] = []
            all_labels[i] = []
            for (inputs, labels) in test_dataLoader[i]:
                all_labels[i].append(labels)
                inputs = Variable(inputs).cuda()
                outputs = model(inputs)
                all_outputs[i].append(outputs.data.cpu())
            all_outputs[i] = torch.cat(all_outputs[i])
            all_labels[i] = torch.cat(all_labels[i])

        all_labels = all_labels[0]

        all_outputs = sum(all_outputs)  # [tensor(), tensor() ,tensor()]

        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()
        _, preds = torch.max(all_outputs, dim=1)

        accuracy = torch.mean((preds == all_labels).float())
        print('TestTTA Acc:%f, Test Total:%d' % (accuracy, len(test_dataset[0])))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def test(model, test_list, is_stacking=False):
    test_dataset = MyDataSet(
        root=test_list,
        transform=preprocess(normalize_torch, IMAGE_SIZE),
        root_pre=IMAGE_PRE
    )
    test_dataLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_dataLoader):
            all_labels.append(labels)
            inputs = Variable(inputs).cuda()

            outputs, _ , _= model(inputs)
            all_outputs.append(outputs.data.cpu())
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

        if is_stacking:
            return

        _, preds = torch.max(all_outputs, dim=1)

        accuracy = torch.mean((preds == all_labels).float())
        print('Test Acc:%f, Test Total:%d' % (accuracy, len(test_dataset)))
        acc = [0 for c in range(38)]
        total = 0
        for c in range(38):
            acc[c] = ((preds == all_labels)*(all_labels == c)).float().sum() / (all_labels == c).sum()
            # print(((preds == all_labels)*(all_labels == c)).float())
            print('%d: %f,%d'%(c, acc[c], (all_labels == c).sum()))
            total = total+acc[c]*(all_labels == c).sum()


def Plant_train(epochNum, path):
    writer = SummaryWriter('log/' + DATE + '/ResNet50/'+path[-10:]+'/')

    IMAGE_PRE_Train = "/raid_new/GPH/DataSet/PlantVillage-Dataset-master/raw/segmented/"

    l_train = []
    test_end = []

    with open('./data/PlantVillage/segmented_sp.json', 'r') as f:
        load_data = json.load(f)
        for t_label in load_data:
            l_train.append(tuple([t_label['image_id'], t_label['disease_class']]))
    random.seed(SEED)
    random.shuffle(l_train)
    lenth = len(l_train)
    divide = int(round(lenth/5))
    test_end = l_train[:divide]
    l_train = l_train[divide:]

    train_dataset = MyDataSet_Sp(
        root=l_train,
        transform=preprocess_baseline_AddBack(normalize_torch, image_size=IMAGE_SIZE),
        root_pre=IMAGE_PRE_Train
    )
    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    # val_dataset = MyDataSet_Sp(
    #     root=test_end,
    #     transform=preprocess(normalize_torch, IMAGE_SIZE),
    #     root_pre=IMAGE_PRE_Train
    # )
    # val_dataLoader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = getModel(path, isTrain=1)

    criterion = models_load.LabelSmoothingCrossEntropy_Weight1().cuda()
    # criterion_sp = nn.CrossEntropyLoss().cuda()
    criterion_sp = models_load.LabelSmoothingCrossEntropy().cuda()
    patience = 0  # 防止局部最小
    lr = 0.0
    min_loss = 100.00
    min_acc = 0.60
    print('min_loss is :%f' % (min_loss))
    for epoch in range(epochNum):
        print('Epoch {}/{}'.format(epoch, epochNum - 1))
        if epoch == 0 or epoch == 1 or epoch == 2:
            lr = 1e-3
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr, amsgrad=True, weight_decay=1e-4)
            # optimizer = torch.optim.SGD(params=model.fresh_params(), lr=lr, momentum=0.9)
        elif epoch == 3:
            lr = 1e-4
            momentum = 0.9
            print('set lr=:%e,momentum=%f' % (lr, momentum))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-4)
        #             optimizer=torch.optim.SGD(params=model.parameters(),lr=lr,momentum=momentum)

        if patience == 3:
            patience = 0
            model.load_state_dict(torch.load('./model/' + MODEL_NAME + '/' + path)['state_dict'])
            print('[+] loading model :' + path)
            lr = lr / 10
            print('loss has increased lr divide 10 lr now is :%e' % (lr))
        running_loss = RunningMean()
        running_corrects_cls = RunningMean()
        running_corrects_sp = RunningMean()
        # disturb = DisturbLabel(alpha=10, C=6)

        prefetcher = DataPrefetcher(train_dataLoader)
        batch = prefetcher.next()
        n_iter = 0
        while batch is not None:
            n_iter += 1

            model.train(True)
            n_batchsize = BATCH_SIZE

            # labels = disturb(labels)

            # 判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
            (inputs, labels) = batch

            # 判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
            if torch.cuda.is_available():
                inputs = Variable(inputs).cuda()
                labels_cls = Variable(labels[0]).cuda()
                labels_sp = Variable(labels[1]).cuda()
            else:
                inputs = Variable(inputs)
                labels_cls = Variable(labels[0])
                labels_sp = Variable(labels[1])
            optimizer.zero_grad()
            outputs_cls, outputs_sp, weight = model(inputs)
            # 找出分数最高的对应的channel，即为top-1类别
            _, preds_cls = torch.max(outputs_cls.data, dim=1)
            _, preds_sp = torch.max(outputs_sp.data, dim=1)

            outputs = []
            outputs.append(outputs_cls)
            outputs.append(weight)

            loss = ALPHA*criterion(outputs, labels_cls)+BETA*criterion_sp(outputs_sp, labels_sp)
            # print(loss.item())
            running_loss.update(loss.item(), 1)
            running_corrects_cls.update(torch.sum(preds_cls == labels_cls.data).data, n_batchsize)
            running_corrects_sp.update(torch.sum(preds_sp == labels_sp.data).data, n_batchsize)

            # running_corrects.update(torch.sum(preds == labels.data).data, n_batchsize)
            loss.backward()
            optimizer.step()
            batch = prefetcher.next()

        ################################################################
        # Log Train
        print('[epoch:%d] :acc_cls: %f,acc_sp: %f,loss:%f,lr:%e,patience:%d' % (
            epoch, running_corrects_cls.value, running_corrects_sp.value, running_loss.value, lr, patience))
        writer.add_scalar('Train/Acc_cls', running_corrects_cls.value, n_iter)
        writer.add_scalar('Train/Acc_sp', running_corrects_sp.value, n_iter)
        writer.add_scalar('Train/Loss', running_loss.value, n_iter)

        # ###############################################################
        # # Log Val
        # lx_cls, px_cls,lx_sp,px_sp = predict(model, val_dataLoader)
        # log_loss = criterion(Variable(px_cls), Variable(lx_cls))
        #
        # log_loss = log_loss.item()
        #
        # _, preds = torch.max(px, dim=1)
        # accuracy = torch.mean((preds == lx).float())
        # print('[epoch:%d]: val_loss:%f,val_acc:%f,' % (epoch, log_loss, accuracy))
        #
        # writer.add_scalar('Val/Acc', accuracy, n_iter)
        # writer.add_scalar('Val/Loss', log_loss, n_iter)
        log_loss = running_loss.value
        if log_loss < min_loss:
            patience = 0
            min_loss = log_loss
            snapshot(os.getcwd() + '/model/' + MODEL_NAME, path, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': log_loss,
                'val_correct': running_corrects_cls.value,
                'val_correct_sp': running_corrects_sp.value
            })
            print('#######################################################################\n'
                  'save new model loss, now loss is ', min_loss)
        else:
            patience += 1
        # if accuracy >= min_acc:
        #     snapshot(os.getcwd() + '/model/' + MODEL_NAME, path.replace('loss', 'acc'), {
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'val_loss': log_loss,
        #         'val_correct': accuracy})
        #     min_acc = accuracy
        #     print('############################################################\n'
        #           'save new model acc,now acc is ', min_acc)
    # export scalar data to JSON for external processing
    writer.export_scalars_to_json('log/' + DATE + '/ResNet50/all_scalars.json')
    writer.close()

def Plant_test():
    l_train = []
    # with open('./data/PlantVillage/color.json', 'r') as f:
    #     load_data = json.load(f)
    #     for t_label in load_data:
    #         path = t_label['image_id']
    #         # path = path.replace('.JPG','_final_masked.jpg')
    #         l_train.append(tuple([path, int(t_label['disease_class']) - 1]))
    # random.seed(SEED)
    # random.shuffle(l_train)
    # lenth = len(l_train)
    # divide = int(round(lenth / 5))
    # test_end = l_train[:divide]

    test_end = []
    with open('./data/PlantVillage/fuza_trim02_Field-PV.json', 'r') as f:
        load_data = json.load(f)
        for t_label in load_data:
            path = t_label['image_id']
            # path = path.replace('.JPG','_final_masked.jpg')
            test_end.append(tuple([path, int(t_label['disease_class']) - 1]))
    test(getModel('2020-01-01_loss_best_plant_basline_noFC_SpLoss_Smooth_l3_11_RCR_Add_WL.pth', isTrain=0), test_end)
    # 2020-01-01_loss_best_plant_basline_noFC_SpLoss_Smooth_l3_11_RCR_Add_WL
    # testTTA(getModel('2019-09-29_loss_best_dropout.pth', isTrain=0), test_end)

if __name__ == '__main__':
    models_load.setup_seed(SEED)
    Plant_train(200, DATE+'_loss_best_plant_BS_B_noRCR.pth')

    # IMAGE_PRE = '/raid_new/GPH/DataSet/PlantVillage-Dataset-master/raw/color/'
    IMAGE_PRE = '/raid_new/GPH/DataSet/PlantVillage-Test - trim02_Field-PV/'
    Plant_test()
