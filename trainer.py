import numpy as np
import torch
import torch.nn as nn
import argparse

from models.lidar_cnn_deep    import LidarCNN_deep, LidarCNN_2_deep, LidarCNN_test
from models.lidar_cnn_shallow import LidarCNN_shallow
# from models.lidar_cnn_2d      import LidarCNN_2D
# from models.lidar_cnn_diff    import LidarCNN_diff

from utils.dataloader import load_LiDARDataset
from utils.evaluation import plot_loss, plot_predictions, plot_mse, plot_multiple_predictions

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate) #filter(lambda p: p.requires_grad, self.model.parameters())
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
   
    path_x =  'data/LiDAR_MovingObstaclesNoRules_old.csv'
    path_y = 'data/risk_MovingObstaclesNoRules_old.csv'

    path_x =  'data/LiDAR_MovingObstaclesNoRules.csv'
    path_y = 'data/risk_MovingObstaclesNoRules.csv'
   
    
    
    data_train, data_val, data_test, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(path_x, path_y, 
                                                                                                            mode='max', 
                                                                                                            batch_size=16, 
                                                                                                            train_test_split=0.7,
                                                                                                            train_val_split=0.3,
                                                                                                            shuffle=True)

    cnn_shallow = LidarCNN_shallow(n_sensors=180, 
                    output_channels=[1],
                    kernel_size=45
                    )

    cnn_deep = LidarCNN_deep(n_sensors=180, 
                    output_channels=[2,4,4,6],
                    kernel_size=9
                    )

    cnn_not_so_deep = LidarCNN_2_deep(n_sensors=180, 
                    output_channels=[3,2,1],
                    kernel_size=45
                    )
    cnn_test = LidarCNN_test(n_sensors=180,
                             output_channels=[3,2,1])

    if args.mode == 'train':
        trainer = Trainer(model=cnn_deep, 
                        epochs=14, 
                        learning_rate=0.0005, 
                        dataloader_train=dataloader_train,
                        dataloader_val=dataloader_val,
                        optimizer='adam')

        trainer.train()

        plot_loss(trainer.training_loss, trainer.validation_loss)
        
        if args.save_model:
            cnn_feature_extractor = cnn_shallow
            print('Saving model')
            torch.save(trainer.model.state_dict(), 'logs/trained_models/model_2_deep_noReLU_pretrained.json')


        trainer.model.eval()
        with torch.no_grad():
            y_pred = trainer.model(data_test.X)

        plot_predictions(y_pred, data_test.y.numpy())

        mse = plot_mse(y_pred, data_test.y.numpy())
        print('mse:', mse)

        
    else:

        # pretrained_dict = torch.load(args.model_path)   
        # model_dict = cnn.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} # 1. filter out unnecessary keys
        # model_dict.update(pretrained_dict) # 2. overwrite entries in the existing state dict
        # cnn.load_state_dict(pretrained_dict) # 3. load the new state dict

        
        cnn_shallow.load_state_dict(torch.load('logs/trained_models/model_shallow_pretrained.json'))
        cnn_not_so_deep.load_state_dict(torch.load('logs/trained_models/model_2_deep_pretrained.json'))
        cnn_deep.load_state_dict(torch.load('logs/trained_models/model_deep_pretrained.json'))

        cnn_shallow.eval()
        cnn_not_so_deep.eval()
        cnn_deep.eval()

        # print(cnn.feature_extractor[0].weight.detach().numpy())
        y_pred = np.zeros((data_test.X.shape[0],3))
        with torch.no_grad():
            y_pred[:,0] = cnn_shallow(data_test.X).numpy().reshape(-1,)
            y_pred[:,1] = cnn_not_so_deep(data_test.X).numpy().reshape(-1,)
            y_pred[:,2] = cnn_deep(data_test.X).numpy().reshape(-1,)

        y_true = data_test.y.numpy()
        
        plot_multiple_predictions(y_pred, y_true, ['1conv','3conv', 'Deep'])
        
        mse_shallow = plot_mse(y_pred[:,0], y_true)
        mse_not_so_deep = plot_mse(y_pred[:,1], y_true)
        mse_deep = plot_mse(y_pred[:,2], y_true)

        # mse_shallow = mean_squared_error(y_true=y_true, y_pred=y_pred[:,0])
        
        # mse_i = np.zeros_like(y_true)
        # for i in range(len(y_true)):
        #     mse_i[i] = mean_squared_error([y_true[i]], [y_pred[i,2]])

        # mse_shallow_std = mse_shallow_std.mean()
        
  
        # print(mse_i.std())
        # print(mse_i.mean())
       
        print('mse_shallow:', mse_shallow)
        print('mse_not_so_deep:', mse_not_so_deep)
        print('mse_deep:', mse_deep)

  

