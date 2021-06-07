import sys
import argparse

import torch

from data import mnist
from model import model1
from torch import nn, optim

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
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
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        parser.add_argument('--epoch', default = 10)

        args = parser.parse_args(sys.argv[2:])
        print(args)
        print('device: ', self.device)
        model = model1().to(self.device)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)    
        
        # TODO: Implement training loop here

        trainloader, _ = mnist()
        train_losses = []

        for e in range(int(args.epoch)):
            print('Epoch: ', e)
            running_loss = 0
            for images, labels in trainloader:
                model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss

            train_losses.append(running_loss)
            print("Train loss:", train_losses[-1].item())

        checkpoint = {'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

        torch.save(checkpoint, 'checkpoint.pth')

        return train_losses

        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = torch.load(args.load_model_from)
        _, test_set = mnist()

        with torch.no_grad():
        # validation pass here
            running_loss = 0
            accuracy = 0
            model.eval()
            for images, labels in testloader:

                log_ps = model(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

                running_loss += loss

            val_losses = running_loss
            accuracy = torch.mean(accuracy)

            print('Val loss: ', val_losses)
            print('Accuracy: ', accuracy)

            return val_losses, accuracy



if __name__ == '__main__':
    TrainOREvaluate()


    
    
    
    
    
    
    
    
    