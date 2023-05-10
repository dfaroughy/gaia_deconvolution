
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from gaia_deconvolution.models.loss import calculate_loss
from gaia_deconvolution.data.plots import plot_data_projections, plot_loss


def Train_Model(model, training_sample, validation_sample, args, show_plots=True, save_best_state=True):        
    train = Train_Epoch(model, args)
    test = Evaluate_Epoch(model, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)
    
    print('INFO: number of training parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    
    for epoch in tqdm(range(args.max_epochs), desc="epochs"):
       
        train.fit(training_sample, optimizer)       
        test.validate(validation_sample)
        scheduler.step() 
       
        print("\t Epoch: {}".format(epoch))
        print("\t Training loss: {}".format(train.loss))
        print("\t Test loss: {}  (min: {})".format(test.loss, test.loss_min))
       
        if test.check_patience(show_plots=show_plots, save_best_state=save_best_state): break
   
    plot_loss(train, test, args)
    torch.cuda.empty_cache()
    return test.best_model


def sampler(model, num_samples, batch_size=10000):
    model.eval()
    with torch.no_grad(): 
        num_batches = num_samples // batch_size + (1 if num_samples % batch_size != 0 else 0)
        samples=[]
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            batch_samples = model.sample(num_samples=current_batch_size)
            samples.append(batch_samples)
        samples = torch.cat(samples, dim=0)
    return samples.cpu().detach()


class Train_Epoch(nn.Module):

    def __init__(self, model, args):
        super(Train_Epoch, self).__init__()
        self.model = model
        self.loss = 0
        self.loss_per_epoch = []
        self.args = args

    def fit(self, data, optimizer):
        
        self.model.train()
        self.loss = 0

        for batch in tqdm(data, desc="batch"):

            if self.args.num_steps <= 1: 
                current_loss = calculate_loss(self.model, batch, self.args)
                current_loss.backward()
                optimizer.step()  
                optimizer.zero_grad()
                self.loss += current_loss.item() / len(data)

            else: 

                # sub-batch and accumulate gradient (use if data does not fit in GPU memory)  

                sub_batches = torch.tensor_split(batch, self.args.num_steps)
                sub_batch_loss = 0
                for sub_batch in sub_batches:
                    current_loss = calculate_loss(self.model, sub_batch, self.args, reduction=torch.sum)
                    current_loss.backward()
                    sub_batch_loss += current_loss.item() / self.args.batch_size
                optimizer.step()
                optimizer.zero_grad()
                self.loss += sub_batch_loss / len(data) 

        self.loss_per_epoch.append(self.loss)

class Evaluate_Epoch(nn.Module):

    def __init__(self, model, args):
        super(Evaluate_Epoch, self).__init__()
        self.model = model
        self.loss = 0
        self.loss_per_epoch = []
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.best_model = None
        self.terminate = False
        self.args = args

    def validate(self, data):
        self.model.eval()
        self.loss = 0
        self.epoch += 1
        for batch in data:

            if self.args.num_steps <= 1: 
                current_loss = calculate_loss(self.model, batch, self.args)
                self.loss += current_loss.item() / len(data)
            else:
                sub_batches = torch.tensor_split(batch, self.args.num_steps)
                sub_batch_loss = 0
                for sub_batch in sub_batches:
                    current_loss = calculate_loss(self.model, sub_batch, self.args, reduction=torch.sum) 
                    sub_batch_loss += current_loss.item() / self.args.batch_size
                self.loss += sub_batch_loss / len(data)
        self.loss_per_epoch.append(self.loss)
        # print('INFO: max GPU memory usage: ', torch.cuda.max_memory_allocated(self.args.device)/1e6, 'MB')

    def check_patience(self, show_plots=True, save_best_state=True):
        self.model.eval()
        if self.loss < self.loss_min:
            self.loss_min = self.loss
            self.patience = 0
            self.best_model = deepcopy(self.model)
            if show_plots:
                with torch.no_grad():
                    sample = sampler(self.best_model, num_samples=20000)
                    sample_x = sample[:,:3]
                    sample_v = sample[:,3:]
                    plot_data_projections(sample_x, label='x', bin_size=0.2, title=r'positions Epoch {}'.format(self.epoch), save_dir=self.args.workdir  + '/result_plots/')
                    plot_data_projections(sample_v, label='v', bin_size=0.2, title=r'velocities Epoch {}'.format(self.epoch), save_dir=self.args.workdir  + '/result_plots/')
            if save_best_state:
                torch.save(self.best_model.state_dict(), self.args.workdir + '/best_model.pth')       
        else: self.patience += 1
        if self.patience >= self.args.max_patience: self.terminate = True
        return self.terminate
