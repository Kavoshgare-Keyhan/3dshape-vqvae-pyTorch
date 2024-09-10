# Import modules
import torch, torchvision, random, csv, h5py, os, yaml, optuna, traceback, logging
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from data_loader import Shapes3DDataset, custom_collate_fn
from vqvae import get_model
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import SubsetRandomSampler, Subset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
from optuna.pruners import MedianPruner  # For early stopping and pruning

os.nice(19)

# Setup logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Introduce mixed precision training with GradScaler
scaler = GradScaler()

# Method to load a pre-trained model (Warm Start / Transfer Learning)
def load_pretrained_model_if_available(model, path='pretrained_vqvae.pth'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        logging.info(f"Loaded pretrained model from {path}")
    else:
        logging.info(f"No pretrained model found at {path}. Training from scratch.")
    return model

# Initialize GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Distributed Training (optional, requires multiple GPUs)
def prepare_distributed_training(model):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training")
    return model

def train_for_one_epoch(epoch_idx, model, data_loader, optimizer, criterion, config):
    r"""
    Method to run the training for one epoch.
    :param epoch_idx: iteration number of current epoch
    :param model: VQVAE model
    :param data_loader: Data loder for 3dshapes
    :param optimizer: optimzier to be used taken from config
    :param criterion: For computing the loss
    :param config: configuration for the current run
    :return:
    """
    recon_losses = []
    codebook_losses = []
    commitment_losses = []
    losses = []

    for im, label in tqdm(data_loader, desc='Training', leave=False): # Ignore the label in DataLoader
        im = im.float().to(device)
        optimizer.zero_grad()
        model_output = model(im)
        output = model_output['generated_image']
        quantize_losses = model_output['quantized_losses']
        recon_loss = criterion(output, im)

        # Mixed precision training
        # with autocast():
        codebook_loss = quantize_losses['codebook_loss'].mean()
        commitment_loss = quantize_losses['commitment_loss'].mean()
        # recon_loss = recon_loss.item()
        loss = (config['train_params']['reconstruction_loss_weight']*recon_loss +
                    config['train_params']['codebook_loss_weight']*codebook_loss +
                    config['train_params']['commitment_loss_weight']*commitment_loss)
        recon_losses.append(recon_loss.item())
        codebook_losses.append(config['train_params']['codebook_loss_weight']*codebook_loss.item())
        commitment_losses.append(commitment_loss.item())
        losses.append(loss.item())
        # Scales the loss, calls backward(), and then unscales gradients
        loss.backward()
        
        # Unscales gradients and calls the step function
        optimizer.step()
        
        
    avg_loss = np.mean(losses)
    logging.info(f"Epoch {epoch_idx + 1} | Loss: {avg_loss:.4f} | Recon Loss : {np.mean(recon_losses):.4f} | Codebook Loss : {np.mean(codebook_losses):.4f} | Commitment Loss : {np.mean(commitment_losses):.4f}")
    return avg_loss

def cross_validate(config, model, dataset, criterion, batch_size):
    ######## Set the desired seed value #######
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Use a simple train-test split instead of full k-fold
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=seed, shuffle=True)

    # Create samples for training and validation set
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Define data loader for train and validation
    train_loader = DataLoader(dataset, batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(dataset, batch_size, sampler=val_sampler, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)

    # Create output directories
    if not os.path.exists(config['train_params']['output_dir']):
        os.mkdir(config['train_params']['output_dir'])

    # set up model params
    num_epochs = 10
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_for_one_epoch(epoch, model, train_loader, optimizer, criterion, config)
        val_loss = validate(model, val_loader, criterion, config)
        
        logging.info(f"Validation Loss after Epoch [{epoch}]: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    return best_val_loss

def validate(model, data_loader, criterion, config): #sample, batch_size
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        loss = 0
        for im, label in data_loader:
            im = im.float().to(device)
            model_output = model(im)
            output = model_output['generated_image']

            recon_loss = criterion(output, im)
            quantize_losses = model_output['quantized_losses']
            codebook_loss = quantize_losses['codebook_loss'].mean()
            commitment_loss = quantize_losses['commitment_loss'].mean()

            loss += (config['train_params']['reconstruction_loss_weight']*recon_loss.item() +
                    config['train_params']['codebook_loss_weight']*quantize_losses['codebook_loss'].item() +
                    config['train_params']['commitment_loss_weight']*quantize_losses['commitment_loss'].item())

    avg_loss = np.mean(loss)
    return avg_loss

 ######## Read the config file #######
config_path='hyperparameters.yaml'
with open(config_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3,log=True)
    config['train_params']['lr'] = learning_rate
    batch_size = 256 #trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    codebook_size = trial.suggest_categorical('codebook_size', [16, 32, 48, 64, 128, 256, 512])
    config['model_params']['codebook_size'] = codebook_size
    beta = trial.suggest_float('beta', 0.1, 1.0,log=True)
    config['train_params']['commitment_loss_weight'] = beta

    # Load a subset of the data for faster tuning
    dataset = Shapes3DDataset(path=config['train_params']['path'])
    subset_indices = list(range(len(dataset) // 5))  # Use 20% of data
    subset = Subset(dataset, subset_indices)

    model = get_model(config).to(device)

    # Apply warm start if possible
    model = load_pretrained_model_if_available(model)
    
    # Prepare for distributed training if multiple GPUs are available
    model = prepare_distributed_training(model)

    criterion = torch.nn.MSELoss()

    # Perform cross-validation
    val_loss = cross_validate(config, model, subset, criterion, batch_size)

    return val_loss

def main():
    try:
        # Use Optuna to find the best hyperparameters with pruning
        study = optuna.create_study(direction='minimize', pruner=MedianPruner())
        study.optimize(objective, n_trials=50)

        logging.info(f"Best hyperparameters: {study.best_params}")

    except Exception as e:
        # Log any errors
        logging.error("An error occurred during the execution.")
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
