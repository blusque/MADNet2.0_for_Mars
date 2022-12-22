import os.path
import random

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from dataset import DEMDataset
from model.generator import Generator
from model.discriminator import Discriminator
from model.loss_function import *
from validate import Validator

import argparse

parser = argparse.ArgumentParser(description="Pytorch MadNet 2.0 for mars")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--beta1", default=0.9, type=float, help="Adam beta 1, Default: 0.9")
parser.add_argument("--beta2", default=0.999, type=float, help="Adam beta 2, Default: 0.999")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

opt = parser.parse_args()
device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")


def main():
    global opt
    print(opt)
    print(device)

    cuda = opt.cuda
    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    gpus = [0, 1, 2]
    opt.seed = random.randint(1, 10000)
    print("Random seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # train_set = DEMDataset("/media/mei/Elements/training_dataset.hdf5")
    train_set = DEMDataset("../../data/training_dataset.hdf5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)

    print("===> Building Model")
    gen_model = Generator().to(device)
    gen_model = DataParallel(gen_model, device_ids=gpus)
    dis_model = Discriminator().to(device)
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

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=>loading model {}".format(opt.pretrained))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            gen_model.load_state_dict(checkpoint['gen_model'].state_dict())
            dis_model.load_state_dict(checkpoint['dis_model'].state_dict())
        else:
            print("=> no model fount at {}".format(opt.pretrained))

    print("===> Setting Optimizer")
    betas = (opt.beta1, opt.beta2)
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=opt.lr, betas=betas)
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=opt.lr, betas=betas)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader,
              (gen_optimizer, dis_optimizer),
              (gen_model, dis_model),
              (g_loss, bh_loss, a_loss),
              epoch
              )
        save_checkpoint(
            (gen_model, dis_model),
            epoch
        )


def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch - 1)
    gen_optimizer, dis_optimizer = optimizer
    g_loss, bh_loss, a_loss = criterion

    for param_group in gen_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in dis_optimizer.param_groups:
        param_group['lr'] = lr

    print("Epoch={}, lr={}".format(epoch, lr))

    gen_model, dis_model = model

    Tensor = torch.cuda.FloatTensor if opt.cuda and torch.cuda.is_available() else torch.FloatTensor
    # print(Tensor)
    err = 0
    rse = 0
    ssim = 0
    dis_loss_val = 0
    gen_loss_val = 0
    writer = SummaryWriter('./log')

    torch.autograd.set_detect_anomaly(True)
    for iteration, batch in enumerate(data_loader, 1):
        if iteration > 100:
            break
        dtm, ori = batch
        dtm = dtm / 255.
        ori = ori / 255.
        # print("ori shape before unsqueezed: ", ori.shape)
        dtm = torch.unsqueeze(dtm, 1)
        ori = torch.unsqueeze(ori, 1)
        # print("ori shape after unsqueezed: ", ori.shape)
        dtm = dtm.to(device)
        ori = ori.to(device)

        valid = Variable(Tensor(dtm.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(dtm.shape[0], 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        gen_optimizer.zero_grad()

        # z = Variable(Tensor(np.random.normal(0, 1, (dtm.shape[0], 1, 512, 512))))

        gen_dtm = gen_model(ori)

        real_predict = dis_model(dtm, ori).detach()
        fake_predict = dis_model(gen_dtm, ori)

        gen_loss = 0.5 * g_loss(dtm, gen_dtm) + 5e-2 * bh_loss(dtm, gen_dtm) \
                   + 5e-3 * a_loss(fake_predict - real_predict.mean(0, keepdim=True), valid)
        # + bh_loss(dtm, gen_dtm) + a_loss(fake_predict
        #                                 - real_predict.mean(0, keepdim=True), valid)

        gen_loss.backward()
        gen_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        dis_optimizer.zero_grad()

        real_predict = dis_model(dtm, ori)
        fake_predict = dis_model(gen_dtm.detach(), ori)

        real_loss = a_loss(real_predict - fake_predict.mean(0, keepdim=True), valid)
        fake_loss = a_loss(fake_predict - real_predict.mean(0, keepdim=True), fake)

        d_loss = (real_loss + fake_loss) / 2

        dis_loss = 0.5 * g_loss(dtm, gen_dtm.detach()) + 5e-2 * bh_loss(dtm, gen_dtm.detach()) + 5e-3 * d_loss
        
        dis_loss.backward()
        dis_optimizer.step()
        
        if iteration % 5 == 0:
            writer.add_images('ground_truth', dtm, epoch * iteration + iteration, dataformats='NCHW')
            writer.add_images('ori', ori, epoch * iteration + iteration, dataformats='NCHW')
            writer.add_images('predicted', gen_dtm, epoch * iteration + iteration, dataformats='NCHW')

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, opt.nEpochs, iteration, len(data_loader), dis_loss.item(), gen_loss.item())
        # )
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.nEpochs, iteration, 10, dis_loss.item(), gen_loss.item())
        )


def save_checkpoint(model, epoch):
    gen_model, dis_model = model
    model_folder = "../checkpoint/"
    gen_model_folder = "../checkpoint/gen/"
    dis_model_folder = "../checkpoint/dis/"
    model_out_path = model_folder + 'model_epoch_{}.pth'.format(epoch)
    gen_model_out_path = gen_model_folder + "gen_model_epoch_{}.pth".format(epoch)
    dis_model_out_path = dis_model_folder + "dis_model_epoch_{}.pth".format(epoch)
    model_state = {'epoch': epoch, 'gen_model': gen_model, 'dis_model': dis_model}
    gen_state = {"epoch": epoch, "model": gen_model}
    dis_state = {"epoch": epoch, "model": dis_model}

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(gen_model_folder):
        os.makedirs(gen_model_folder)
    if not os.path.exists(dis_model_folder):
        os.makedirs(dis_model_folder)

    torch.save(model_state, model_out_path)
    torch.save(gen_state, gen_model_out_path)
    torch.save(dis_state, dis_model_out_path)

    print("Checkpoint saved to {} & {}".format(gen_model_out_path, dis_model_out_path))


if __name__ == '__main__':
    main()
