# Import modules
import torch, torchvision, random, os, yaml, logging
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from data_loader import Shapes3DDataset, custom_collate_fn
from vqvae import get_model
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

# Setup logging
def setup_logger(config):
    # Create output directories
    if not os.path.exists(config['train_params']['output_dir']): os.mkdir(config['train_params']['output_dir'])
    log_path = os.path.join(config['train_params']['output_dir'], config['train_params']['model_name'])
    logging.basicConfig(filename='{log_path}.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Method to load a pre-trained model (Warm Start / Transfer Learning)
def load_pretrained_model_if_available(model, config):
    path=os.path.join(config['train_params']['output_dir'], config['train_params']['model_name'])
    try:
        model.load_state_dict(torch.load(path))
        return model
    except:
        logging.error(f"No saved model with the given name found at the specified path: {path}", exc_info=True)

# Distributed Training (optional, requires multiple GPUs)
def prepare_distributed_training(model):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training")
    return model

def split_dataset(dataset, config, test_size=0.2):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    # Split data indices randomly into two groups
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=config['train_params']['seed'], shuffle=True)
    # Create train and test dataset 
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset

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
        # Take average on quantize_losses tensor elements due to paralle/distributed run on GPU
        codebook_loss = quantize_losses['codebook_loss'].mean()
        commitment_loss = quantize_losses['commitment_loss'].mean()
        # compute total loss for each image
        loss = (config['train_params']['reconstruction_loss_weight']*recon_loss +
                    config['train_params']['codebook_loss_weight']*codebook_loss +
                    config['train_params']['commitment_loss_weight']*commitment_loss)
        recon_losses.append(recon_loss.item())
        codebook_losses.append(config['train_params']['codebook_loss_weight']*codebook_loss.item())
        commitment_losses.append(commitment_loss.item())
        losses.append(loss.item())
        # call backward and optimize loss function
        loss.backward()
        optimizer.step()
        
        
    avg_loss = np.mean(losses)
    logging.info(f'Finished epoch: {epoch_idx + 1} |
                Total Loss: {avg_loss:.4f} |
                Recon Loss : {np.mean(recon_losses):.4f} | 
                Codebook Loss : {np.mean(codebook_losses):.4f} | 
                Commitment Loss : {np.mean(commitment_losses):.4f}')
    return avg_loss

def train(config, model, dataset, criterion, batch_size=256, save_option=True):
    ######## Set the desired seed value #######
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Define data loader for train
    train_loader = DataLoader(dataset, batch_size, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)

    # set up model params
    num_epochs = config['train_params']['epochs']
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    best_loss = np.inf

    for epoch in range(num_epochs):
        train_loss = train_for_one_epoch(epoch, model, train_loader, optimizer, criterion, config)
        if train_loss < best_loss:
            logging.info(f'Improved Loss to {train_loss:.4f} .... Saving Model')
            best_loss = train_loss
        else:
            logging.info(f'No Loss Improvement')
        
    # Save the final model and codebook indices
    if save_option:
        all_indices = []
        with torch.no_grad():
            for im, label in tqdm(train_loader, desc='Recall trained model to save codebook indices', leave=False): # Ignore the label in DataLoader
                im = im.float().to(device)
                model_output = model(im)
                indices = model_output['quantized_indices']
                all_indices.append(indices.cpu())

        # Concatenate all indices into a single tensor and save it
        indices_tensor = torch.cat(all_indices, dim=0)
        torch.save(indices_tensor, os.path.join(config['train_params']['output_dir'], config['train_params']['indices_tensor']))
        # Save the model with provided path and name
        torch.save(model.state_dict(), os.path.join(config['train_params']['output_dir'], config['train_params']['model_name']))

# Define a new function to apply trained model to the unseen data which we extract it from original dataset
def reconstruction_acc(config, model, dataset, criterion, batch_size=256):
    # Define data loader for train
    test_loader = DataLoader(dataset, batch_size, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)

    # Load the trained model
    model = load_pretrained_model_if_available(model, config)
    model.eval()
    with torch.no_grad():
        total_recon_loss = 0
        total_codebook_loss = 0
        total_commitment_loss = 0
        total_loss = 0
        for im, label in test_loader:
            im = im.float().to(device)
            model_output = model(im)
            output = model_output['generated_image']

            recon_loss = criterion(output, im)
            quantize_losses = model_output['quantized_losses']

            total_recon_loss += recon_loss.item()
            total_codebook_loss += quantize_losses['codebook_loss'].mean().item()
            total_commitment_loss += quantize_losses['commitment_loss'].mean().item()
            total_loss += (config['train_params']['reconstruction_loss_weight']*recon_loss.item() +
                        config['train_params']['codebook_loss_weight']*quantize_losses['codebook_loss'].mean().item() +
                        config['train_params']['commitment_loss_weight']*quantize_losses['commitment_loss'].mean().item())

        avg_recon_loss = np.mean(total_recon_loss)
        avg_codebook_loss = np.mean(total_codebook_loss)
        avg_commitment_loss = np.mean(total_commitment_loss)
        avg_loss = np.mean(total_loss)
        logging.info(f'Total Test Loss : {avg_loss:.4f} | Recon Loss : {avg_recon_loss:.4f} | Codebook Loss : {avg_codebook_loss:.4f} | Commitment Loss : {avg_commitment_loss:.4f}')

    return avg_loss

def reconstruction_img(config, model, dataset, save_option=True):
    # Sample images for reconsruction and comparisons
    idxs = torch.randint(0, len(dataset), (100, )) # Randomly sample indices for reconstruction.
    ims = torch.cat([dataset[idx][0][None, :] for idx in idxs]).float() # Create a batch of images from the test set using sampled indices.
    ims = ims.to(device)

    # Load the trained model 
    model = load_pretrained_model_if_available(model, config)
    model_output = model(ims) # Generate reconstructed images.
    generated_ims = model_output['generated_image']
    ims = (ims+1)/2
    generated_ims = (generated_ims + 1) / 2  # Normalize to [0, 1] for visualization

    ## Transform to original value to retrieve colorized data
    ims = ims * 255.0  # Scale to [0, 255]
    generated_ims = generated_ims * 255.0  # Scale to [0, 255]
    ims = ims.cpu().numpy().astype(np.uint8)
    generated_ims = generated_ims.detach().cpu().numpy().astype(np.uint8)  # Detach before converting to NumPy

    # combined_images = torch.hstack([ims, generated_im])
    combined_images = np.concatenate((ims, generated_ims), axis=3)

    # Rearrange to a grid
    combined_images = torch.tensor(combined_images).permute(0, 1, 2, 3)  # Change to [batch, channels, height, width]

    grid = torchvision.utils.make_grid(combined_images, nrow=10, padding=2)
    img = torchvision.transforms.ToPILImage()(grid)
    if save_option: img.save('reconstruction_{model_name}.png'.format(model_name=config['train_params']['model_name']))

    # plot the rexonstructions in a large figure
    # plt.figure(figsize=(20, 10))
    # plt.imshow(img)

def train_vqvae(config_path='hyperparameters.yaml'):
    # Read the hyperparameters
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            logging.error(f"{e}", exc_info=True)
    
    # Load data as a dataset and split into training and test datasets
    dataset = Shapes3DDataset(path=config['train_params']['path'])
    train_dataset, test_dataset = split_dataset(dataset, config)

    # Initialize model
    model = get_model(config).to(device)

    # Assessment metric
    criterion = {
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss()
    }.get(config['train_params']['crit'])

    # Train model
    train(config, model, train_dataset, criterion)
    # Check accuracy of the trained model by applying it to unseen data
    reconstruction_acc(config, model, dataset, criterion)
    # Reconstruct images
    reconstruction_img(config, model, test_dataset)


if __name__ == '__main__':
    # The niceness value is a number that affects the priority of a process. The lower the niceness value, the higher the priority of the process, meaning it will get more CPU time
    os.nice(10)
    # Initialize processing hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instanciate logging
    setup_logger()
    train_vqvae()

