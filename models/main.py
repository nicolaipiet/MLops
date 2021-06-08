import sys
import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import wandb

from models.model import MyAwesomeModel
from src.data.make_dataset import mnist


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter()

        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>")
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.003)
        parser.add_argument('--epoch', default=10)
        parser.add_argument('--model_version', default=00)
        parser.add_argument('--log_interval', default=100)

        args = parser.parse_args(sys.argv[2:])
        print(args)
        print('device: ', self.device)

        wandb.init(config=args)

        model = MyAwesomeModel().to(self.device)
        wandb.watch(model, log_freq=int(args.log_interval))

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

        trainset, _ = mnist()
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=64,
                                                  shuffle=True)
        train_losses = []

        for e in range(int(args.epoch)):
            print('Epoch: ', e)
            running_loss = 0
            log_ps_lst = []
            for batch_idx, (images, labels) in enumerate(trainloader):
                model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss
                log_ps_lst.append(log_ps.flatten())

                if batch_idx % args.log_interval == 0:
                    wandb.log({
                        "loss":
                        loss,
                        "MNIST examples":
                        wandb.Image(images.view(64, 1, 28, 28)[0, 0, :, :])
                    })

            train_losses.append(running_loss)

            self.writer.add_scalar('loss/train', running_loss, e)
            self.writer.add_histogram('Class probability distribution',
                                      torch.cat(log_ps_lst), e)

            print("Train loss:", train_losses[-1].item())

        self.writer.add_hparams(
            {
                'lr': float(args.lr),
                'epochs': int(args.epoch)
            }, {'hparam/loss': train_losses[-1].item()})

        torch.save(
            model.state_dict(), 'models/trained_models/' +
            str(args.model_version) + '_checkpoint.pth')

        fig = plt.figure()
        x = np.arange(int(args.epoch))
        plt.plot(x, train_losses, label='Train loss')
        plt.legend(loc='upper right')
        plt.savefig('reports/figures/train_loss.png')

        self.writer.close()
        wandb.finish()
        return train_losses

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        if args.load_model_from:
            state_dict = torch.load(args.load_model_from)

        model = MyAwesomeModel().to(self.device)
        model.load_state_dict(state_dict)

        _, testset = mnist()
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=64,
                                                 shuffle=False)

        with torch.no_grad():
            # validation pass here
            running_loss = 0
            accuracy = 0
            model.eval()
            for images, labels in testloader:

                log_ps = model(images.to(self.device))
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape).to(
                    self.device)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            accuracy = accuracy / len(testloader)
            print('Accuracy: ', accuracy.item())

            return accuracy


if __name__ == '__main__':
    TrainOREvaluate()
