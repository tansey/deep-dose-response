############################################################
'''Tools for fitting deep learning models'''
############################################################
import numpy as np
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from utils import create_folds, batches, clip_gradient

'''Wrapper neural regression model that standardizes inputs and outputs'''
class StandardizedNeuralModel():
    def __init__(self, model, X_means, X_stds):
        self.X_means = X_means
        self.X_stds = X_stds
        self.nfeatures = X_means.shape[0]
        self.model = model

    def predict(self, X):
        self.model.eval()
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means[None]) / self.X_stds[None]), requires_grad=False)
        tYhat = self.model(tX)
        return tYhat.data.numpy()

def fit(X, model_fn, loss_fn,
           nepochs=100, val_pct=0.1,
           batch_size=10,
           verbose=False, lr=0.1, optimizer='RMSprop',
           weight_decay=1e-4, momentum=0.9, step_decay=0.9998,
           file_checkpoints=True, checkpoint_file=None, save_checkpoint=False,
           indices=None,
           **kwargs):
    if file_checkpoints and checkpoint_file is None:
        import uuid
        checkpoint_file = '/tmp/tmp_file_' + str(uuid.uuid4())

    if indices is None:
        # Create train/validate splits
        indices = np.arange(X.shape[0], dtype=int)
        np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices = indices[:train_cutoff]
        validate_indices = indices[train_cutoff:]
    else:
        train_indices, validate_indices = indices
    X_train = X[train_indices]
    X_val = X[validate_indices]

    # Standardize the features (helps with gradient propagation)
    X_std = X_train.std(axis=0)
    X_mean = X_train.mean(axis=0)
    X_std[X_std == 0] = 1 # Handle constant features
    tX = autograd.Variable(torch.FloatTensor((X - X_mean[None]) / X_std[None]), requires_grad=False)
    
    model = model_fn()

    # Save the model to file
    if file_checkpoints:
        torch.save(model, checkpoint_file)
    else:
        import pickle
        model_str = pickle.dumps(model)

    # Setup the SGD method
    print('Using lr={}'.format(lr))
    if optimizer == 'SGD':
        optimizer_obj = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, nesterov=True, momentum=momentum)
    elif optimizer == 'RMSprop':
        optimizer_obj = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer_obj = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer))
    scheduler = optim.lr_scheduler.StepLR(optimizer_obj, step_size=1, gamma=step_decay)

    # Track progress
    train_losses, val_losses, best_loss = np.zeros(nepochs), np.zeros(nepochs), None
    num_bad_epochs = 0
    
    # Train the model
    for epoch in range(nepochs):
        if verbose:
            print('\t\tEpoch {}'.format(epoch+1), flush=True)

        # Track the loss curves
        train_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=True)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tBatch {}'.format(batch_idx))
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to training mode
            model.train()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predictions
            tOut = model(tX[tidx])

            # Calculate the loss
            loss = loss_fn(tidx, tOut, train=True).mean()
            # print(loss)

            # Calculate gradients
            loss.backward()

            # Pre-clipping
            # print('----------------------------- PRE clipping -------------------------')
            # for p in model.parameters():
            #     print('===========\ngradient ({}):{}\n----------\n{}\n----------\nValues Min: {} Max: {}\nGradients Min: {} Max: {}'.format(p.grad.shape,p.data,p.grad,p.data.min(),p.data.max(),p.grad.min(), p.grad.max()))

            # Clip the gradients
            # clip_gradient(model)
            # print('----------------------------- POST clipping -------------------------')
            # for p in model.parameters():
            #     print('===========\ngradient ({}):{}\n----------\n{}\n----------\nValues Min: {} Max: {}\nGradients Min: {} Max: {}'.format(p.grad.shape,p.data,p.grad,p.data.min(),p.data.max(),p.grad.min(), p.grad.max()))


            # Apply the update
            # [p for p in model.parameters() if p.requires_grad]
            optimizer_obj.step()

            # Track the loss
            train_loss += loss.data

        validate_loss = torch.Tensor([0])
        for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
            if verbose and (batch_idx % 100 == 0):
                print('\t\t\tValidation Batch {}'.format(batch_idx), flush=True)
            tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

            # Set the model to test mode
            model.eval()

            # Reset the gradient
            model.zero_grad()

            # Run the model and get the predictions
            tOut = model(tX[tidx])

            # Calculate the loss
            loss = loss_fn(tidx, tOut).sum()

            # Track the loss
            validate_loss += loss.data

        train_losses[epoch] = train_loss.numpy() / float(len(train_indices))
        val_losses[epoch] = validate_loss.numpy() / float(len(validate_indices))

        # Adjust the learning rate down if the validation performance is bad
        scheduler.step(val_losses[epoch])

        if np.isnan(val_losses[epoch]):
            if verbose:
                print('\t\tNetwork went to NaN. Readjusting learning rate down by 50%', flush=True)
            if file_checkpoints:
                os.remove(checkpoint_file)
            return fit(X, model_fn, loss_fn,
                        nepochs=nepochs, val_pct=val_pct,
                        batch_size=batch_size,
                        verbose=verbose, lr=lr*0.5,
                        weight_decay=weight_decay, momentum=momentum,
                        optimizer=optimizer, step_decay=step_decay,
                        file_checkpoints=file_checkpoints,
                        checkpoint_file=checkpoint_file,
                        save_checkpoint=save_checkpoint,
                        indices=(train_indices, validate_indices),
                        **kwargs)

        # Check if we are currently have the best held-out log-likelihood
        if epoch == 0 or val_losses[epoch] <= best_loss:
            if verbose:
                print('\t\t\tSaving test set results.      <----- New high water mark on epoch {}'.format(epoch+1), flush=True)
            # If so, use the current model on the test set
            best_loss = val_losses[epoch]
            if file_checkpoints:
                torch.save(model, checkpoint_file)
            else:
                import pickle
                model_str = pickle.dumps(model)
        else:
            num_bad_epochs += 1
        
        if verbose:
            print('Validation loss: {} Best: {}'.format(val_losses[epoch], best_loss), flush=True)

    if file_checkpoints:
        model = torch.load(checkpoint_file)
        if not save_checkpoint:
            os.remove(checkpoint_file)
    else:
        import pickle
        model = pickle.loads(model_str)
    return {'model': StandardizedNeuralModel(model, X_mean, X_std),
            'train': train_indices,
            'validation': validate_indices,
            'train_loss': train_losses,
            'validation_loss': val_losses}








