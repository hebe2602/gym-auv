import numpy as np
import torch
import torch.nn as nn
import argparse
import sys 

from model.network import LidarCNN
from utils.dataloader import load_LiDARDataset
from utils.evaluation import plot_loss, plot_predictions, plot_mse



class Trainer():
    def __init__(self, 
                 model:nn.Module,
                 epochs:int,
                 learning_rate:float,
                 dataloader_train:torch.utils.data.DataLoader,
                 dataloader_val:torch.utils.data.DataLoader,
                 optimizer:str = 'sgd'   
                 ) -> None:

        self.model            = model
        self.epochs           = epochs
        self.learning_rate    = learning_rate
        self.dataloader_train = dataloader_train
        self.dataloader_val   = dataloader_val
        self.loss             = nn.MSELoss()
        
        if optimizer == 'adam':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer     = torch.optim.SGD(self.model.parameters(),  lr=self.learning_rate, momentum=0.9)

        self.training_loss = []
        self.validation_loss = []
    
    
    def train_batch(self, X_batch, Y_batch) -> float:
 
        Y_pred = self.model(X_batch)              # Do forward pass
        loss   = self.loss(Y_pred, Y_batch)       # Calculate loss

        loss.backward()                           # Perform backward pass, i. e. calcualte gradients
        self.optimizer.step()                     # Update parameters
        self.optimizer.zero_grad()                # Reset computed gradients

        return loss.item()
    
    def validation_step(self):
        self.model.eval()  # Deactivates certain layers such as dropout and batch normalization

        with torch.no_grad():
            num_batches = 0
            avg_loss = 0
            for X_batch, y_batch in self.dataloader_val: 
                y_pred = self.model(X_batch)
                val_loss = self.loss(y_pred, y_batch)
                num_batches += 1
                avg_loss += val_loss
        avg_loss = avg_loss/num_batches
        
        self.model.train()
        return  avg_loss.item() #val_loss.item()

    def train(self):
        step = 0
        print('---------')
        print('Training')
        print('---------')
        for e in range(self.epochs):
            train_loss_epoch = 0
            num_batches = 0

            for X_batch, y_batch in self.dataloader_train: 
                train_loss = self.train_batch(X_batch, y_batch)
                train_loss_epoch += train_loss
                num_batches += 1
        
            val_loss   = self.validation_step()

            self.training_loss.append(train_loss_epoch/num_batches)
            self.validation_loss.append(val_loss)
            
            print('EPOCH', e+1, ': \tTraining loss:', train_loss_epoch/num_batches, '\tValidation loss:', val_loss)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        help='Which mode to run the program',
        choices=['train', 'evaluate']
    )
    parser.add_argument(
        '--model_path',
        help='Path with model to evaluate',
    )
    parser.add_argument(
        '--save_model',
        help='',
        default=False
    )
    args = parser.parse_args()

   
    torch.manual_seed(2)
    # path_x = 'data/LiDAR.txt' 
    # path_y = 'data/risks.txt' 
    path_x =  'data/LiDAR_moving_obstacles.txt'
    path_y = 'data/risks_moving_obstacles.txt'

    data_train, data_val, data_test, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(path_x, path_y, 
                                                                                                            mode='sum', 
                                                                                                            batch_size=16, 
                                                                                                            train_test_split=0.7,
                                                                                                            shuffle=True)
      
    cnn = LidarCNN(n_sensors=180, 
                        output_channels=[4,4, 4],
                        kernel_size=9
                        )


    if args.mode == 'train':
        trainer = Trainer(model=cnn, 
                        epochs=11, 
                        learning_rate=0.001, 
                        dataloader_train=dataloader_train,
                        dataloader_val=dataloader_test,
                        optimizer='adam')

        trainer.train()
        plot_loss(trainer.training_loss, trainer.validation_loss)
        
        if args.save_model:
            print('Saving model')
            torch.save(trainer.model.state_dict(), 'logs/trained_models/model_2.json')


        trainer.model.eval()
        with torch.no_grad():
            y_pred = trainer.model(data_test.X)
            y_train_pred = trainer.model(data_train.X)
        trainer.model.eval()

        plot_predictions(y_pred, data_test.y.numpy())
        plot_predictions(y_train_pred, data_train.y.numpy())
        mse = plot_mse(y_pred, data_test.y.numpy())
        print(mse)
    
    else:
        cnn.load_state_dict(torch.load(args.model_path))
        print('Loading model')
        print(cnn)
        cnn.eval()
        with torch.no_grad():
            y_pred = cnn(data_test.X)

        plot_predictions(y_pred, data_test.y.numpy())


