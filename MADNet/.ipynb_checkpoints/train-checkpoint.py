import os.path
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time

from dataset import DEMDataset
from model.generator import Generator
from model.discriminator import Discriminator
from model.loss_function import *
from validate import Validator
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Pytorch MadNet 2.0 for mars")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--gen-lr", type=float, default=1e-4, help="Generator Learning Rate. Default=1e-3")
parser.add_argument("--dis-lr", type=float, default=5e-7, help="Discriminator Learning Rate. Default=1e-5")
parser.add_argument("--gen-step", type=int, default=200)
parser.add_argument("--dis-step", type=int, default=200,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--beta1", default=0.9, type=float, help="Adam beta 1, Default: 0.9")
parser.add_argument("--beta2", default=0.999, type=float, help="Adam beta 2, Default: 0.999")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--save-per-epochs", "-p", default=10, type=int, help="How many epochs the checkpoint is saved once.")
parser.add_argument("--dataset", "-d", default="", type=str, help="Path to Dataset")

opt = parser.parse_args()
device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
rse_data = []
ssim_data = []
epoch_data = []


def main():
    global opt
    print(opt)
    print(device)

    cuda = opt.cuda
    if os.name == "nt":
        opt.threads = 0
    if cuda and torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    elif cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    gpus = [0]
    opt.seed = random.randint(1, 10000)
    print("Random seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = None
    if os.name == "nt":
        train_set = DEMDataset(opt.dataset)
    elif os.name == "posix":
        train_set = DEMDataset(opt.dataset)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=False, drop_last=True)

    print("===> Building Model")
    gen_model = Generator().to(device)
    dis_model = Discriminator().to(device)
    if cuda:
        gen_model = DataParallel(gen_model, device_ids=gpus)
        dis_model = DataParallel(dis_model, device_ids=gpus)
    g_loss = GradientLoss().to(device)
    bh_loss = BerhuLoss().to(device)
    a_loss = torch.nn.BCEWithLogitsLoss().to(device)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=>loading checkpoint {}".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            gen_model.load_state_dict(checkpoint['gen_model'].state_dict())
            dis_model.load_state_dict(checkpoint['dis_model'].state_dict())
        else:
            print("=> no checkpoint fount at {}".format(opt.resume))

    print("===> Setting Optimizer")
    betas = (opt.beta1, opt.beta2)
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=opt.gen_lr, betas=betas)
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=opt.dis_lr, betas=betas)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.start_epoch + opt.nEpochs):
        train(training_data_loader,
              (gen_optimizer, dis_optimizer),
              (gen_model, dis_model),
              (g_loss, bh_loss, a_loss),
              epoch)
        if epoch % opt.save_per_epochs == 0:
            save_checkpoint((gen_model, dis_model),epoch)


def adjust_learning_rate(epoch, type: str):
    global opt
    if type == 'gen':
        lr = opt.gen_lr * (0.1 ** ((epoch - opt.start_epoch + 1) // opt.gen_step))
    elif type == 'dis':
        lr = opt.dis_lr * (0.1 ** ((epoch - opt.start_epoch + 1) // opt.dis_step))
    return lr


def train(data_loader, optimizer, model, criterion, epoch):
    global rse_data, ssim_data, epoch_data, opt
    gen_lr = adjust_learning_rate(epoch - 1, 'gen')
    dis_lr = adjust_learning_rate(epoch - 1, 'dis')
    gen_optimizer, dis_optimizer = optimizer
    g_loss, bh_loss, a_loss = criterion

    for param_group in gen_optimizer.param_groups:
        param_group['lr'] = gen_lr
    for param_group in dis_optimizer.param_groups:
        param_group['lr'] = dis_lr

    print("[Epoch %d/%d] " % (epoch, opt.nEpochs + opt.start_epoch - 1),
          "gen_lr={}, dis_lr={}".format(gen_lr, dis_lr))

    gen_model, dis_model = model

    Tensor = torch.cuda.FloatTensor if opt.cuda and torch.cuda.is_available() else torch.FloatTensor
    
    writer = SummaryWriter('../log')
    if not os.path.exists('../img'):
        os.mkdir('../img')
    torch.autograd.set_detect_anomaly(True)
    
    total_rse = 0.
    total_ssim = 0.
    bar = tqdm(enumerate(data_loader, 1), leave=False, total=len(data_loader))
    bar.set_description('Iteration ' + str(0))
    bar.set_postfix(
        D_loss=None, 
        G_loss=None,
        gd_loss=None,
        bh_loss=None,
        a_loss=None)
    sample_time = 0
    for iteration, batch in bar:
        start0 = time.time()
        dtm, ori = batch
        dtm = dtm / 255.
        ori = ori / 255.
        dtm = torch.unsqueeze(dtm, 1)
        ori = torch.unsqueeze(ori, 1)
        dtm = dtm.to(device)
        ori = ori.to(device)
        valid = Variable(Tensor(dtm.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(dtm.shape[0], 1).fill_(0.0), requires_grad=False)
        gen_dtm = gen_model(ori)
        # ---------------------
        #  Train Discriminator
        # ---------------------

        dis_optimizer.zero_grad()
        
        real_predict = dis_model(dtm, ori)
        fake_predict = dis_model(gen_dtm.detach(), ori)
        real_loss = a_loss(real_predict - fake_predict.mean(0, keepdim=True), valid)
        fake_loss = a_loss(fake_predict - real_predict.mean(0, keepdim=True), fake)

        dis_loss = (real_loss + fake_loss) / 2
        
        dis_loss.backward()
        dis_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------

        gen_optimizer.zero_grad()

        # z = Variable(Tensor(np.random.normal(0, 1, (dtm.shape[0], 1, 512, 512))))

        real_predict = dis_model(dtm, ori).detach()
        fake_predict = dis_model(gen_dtm, ori)
        real_loss = a_loss(real_predict - fake_predict.mean(0, keepdim=True), fake)
        fake_loss = a_loss(fake_predict - real_predict.mean(0, keepdim=True), valid)
        
        g_loss_value = g_loss(dtm, gen_dtm)
        bh_loss_value = bh_loss(dtm, gen_dtm)
        # print('g_loss: {}, bh_loss: {}, a_loss: {}'.format(g_loss_value, bh_loss_value
        #                                                    , (real_loss + fake_loss) / 2))
        gen_loss = 500 * g_loss_value + 0.5 * bh_loss_value \
                   + 5e-2 * (real_loss + fake_loss) / 2

        gen_loss.backward()
        gen_optimizer.step()
        
        if iteration % 100 == 0:
            sample_time += 1
            np_dtm = dtm.cpu().detach().numpy()
            np_gen_dtm = gen_dtm.cpu().detach().numpy()
            val = Validator(np_dtm, np_gen_dtm)
            rse, ssim = val.validate()
            step = (epoch - opt.start_epoch) * (len(data_loader) // 100) + sample_time
            writer.add_scalar('rse', rse, step)
            writer.add_scalar('ssim', ssim, step)
            
        if iteration == len(data_loader):
            writer.add_images('ground_truth', dtm, epoch, dataformats='NCHW')
            writer.add_images('ori', ori, epoch, dataformats='NCHW')
            writer.add_images('predicted', gen_dtm, epoch, dataformats='NCHW')
            origin = ori.cpu().detach().numpy()[0, 0, ...]
            result = gen_dtm.cpu().detach().numpy()[0, 0, ...]
            save = np.concatenate((origin, result), axis=0)
            plt.imsave(f'../img/epoch{epoch}.png', save)
            
        bar.set_description('Iteration ' + str(iteration))
        bar.set_postfix(
            D_loss=dis_loss.item(), 
            G_loss=gen_loss.item(),
            gd_loss=g_loss_value.item(),
            bh_loss=bh_loss_value.item(),
            a_loss=((real_loss + fake_loss) / 2).item())
    print(
        "[Epoch %d/%d]"
        % (epoch, opt.nEpochs + opt.start_epoch - 1))


def save_checkpoint(model, epoch):
    gen_model, dis_model = model
    model_folder = "../checkpoint/"
    gen_model_folder = "../checkpoint/gen/"
    model_out_path = model_folder + 'model_epoch_{}.pth'.format(epoch)
    gen_model_out_path = gen_model_folder + "gen_model_epoch_{}.pth".format(epoch)
    model_state = {'epoch': epoch, 'gen_model': gen_model, 'dis_model': dis_model}
    gen_state = {"epoch": epoch, "model": gen_model}

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(gen_model_folder):
        os.makedirs(gen_model_folder)

    torch.save(model_state, model_out_path)
    torch.save(gen_state, gen_model_out_path)

    print("Checkpoint saved to {} & {}".format(gen_model_out_path, model_out_path))


if __name__ == '__main__':
    main()
