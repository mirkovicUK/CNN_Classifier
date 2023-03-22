import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.optim as optim


def lr_finder(min_lr, max_lr, n_steps, loss, model, data_loaders):
    
    if torch.cuda.is_available():
        model.cuda()
    
    # Save initial weights so we can restore them at the end
    torch.save(model.state_dict(), "__weights_backup")
    
    # specify optimizer
    optimizer = optim.SGD(model.parameters(), lr=min_lr)

    # We create a learning rate scheduler that increases the learning
    # rate at every batch.
    # Find the factor where min_lr * r**(n_steps-1) = max_lr
    r = np.power(max_lr / min_lr, 1 / (n_steps - 1))

    def new_lr(epoch):
        """
        This should return the *factor* by which the initial learning
        rate must be multipled for to get the desired learning rate
        """
        return r ** epoch

    # This scheduler increases the learning rate by a constanct factor (r)
    # at every iteration
    lr_scheduler = LambdaLR(optimizer, new_lr)

    # Set the model in training mode
    # (so all layers that behave differently between training and evaluation,
    # like batchnorm and dropout, will select their training behavior)
    model.train()

    # Loop over the training data
    losses = {}
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(data_loaders["train"]),
        desc="Training",
        total=len(data_loaders["train"]),
        leave=True,
        ncols=80,
    ):
        # move data to GPU if available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad()  # -
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)  # =
        # 3. calculate the loss
        loss_value = loss(output, target)  # =
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()  # -
        # 5. perform a single optimization step (parameter update)
        optimizer.step()  # -

        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        ) 


        losses[lr_scheduler.get_last_lr()[0]] = train_loss

        # Stop if the loss gets too big
        if train_loss / min(losses.values()) > 10:
            break

        if batch_idx == n_steps - 1:
            break
        else:
            # Increase the learning rate for the next iteration
            lr_scheduler.step()
    
    # Restore model to its initial state
    model.load_state_dict(torch.load('__weights_backup'))

    return losses