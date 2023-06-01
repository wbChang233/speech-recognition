import argparse
import warnings
import srnet as srnet
from loss import *
import torch.optim as optim
from data_processor import *
from trainer import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


def main():
    global args
    args = parser.parse_args()

    if args.model == 'srnet':
        model = srnet.SRNet()
        criterion = Prediction_loss
        args.save_dir = 'trained_srnet'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        torch.backends.cudnn.benchmark = True

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [35, 70], gamma=0.25, last_epoch=-1)
    train_loader, valid_loader = data_loader(args)

    print('\nModel: %s\n'
          'Epoch: %d\n'
          'Initial LR: %f\n'
          % (args.model, args.epochs, args.lr))

    print('Start training')
    for epoch in range(args.epochs):
        print('\ncurrent lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        if args.model == 'srnet':
            loss = train(train_loader, model, criterion, optimizer, epoch)
            scheduler.step()
            top1_Accuracy = valid(valid_loader, model)
            print(
                f"---loss: {loss}---top1:{top1_Accuracy}%")

    torch.save(model.state_dict(), os.path.join(args.save_dir, f"/{args.model}_{args.epochs}.pth"))
    print('Trained finished.')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='srnet',
                        choices=['srnet'],
                        help='choose model to train')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained model',
                        default='trained_srnet', type=str)

    main()
