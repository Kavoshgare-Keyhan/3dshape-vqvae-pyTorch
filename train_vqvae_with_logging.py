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

# Setup logging
logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        # if config['train_params']['save_training_image']:
        #     cv2.imwrite('input.jpeg', (255 * (im.detach() + 1) / 2).cpu().permute(0, 1, 2, 3).numpy().astype(np.uint8)) #(255 * (im.detach() + 1) / 2).cpu().permute((0, 2, 3, 1)).numpy()[0]
        #     cv2.imwrite('output.jpeg', (255 * (output.detach() + 1) / 2).cpu().permute(0, 1, 2, 3).numpy().astype(np.uint8)) #(255 * (output.detach() + 1) / 2).cpu().permute((0, 2, 3, 1)).numpy()[0]

        recon_loss = criterion(output, im)
        loss = (config['train_params']['reconstruction_loss_weight']*recon_loss +
                config['train_params']['codebook_loss_weight']*quantize_losses['codebook_loss'] +
                config['train_params']['commitment_loss_weight']*quantize_losses['commitment_loss'])
        recon_losses.append(recon_loss.item())
        codebook_losses.append(config['train_params']['codebook_loss_weight']*quantize_losses['codebook_loss'].item())
        commitment_losses.append(quantize_losses['commitment_loss'].item())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    print('Finished epoch: {} | Recon Loss : {:.4f} | Codebook Loss : {:.4f} | Commitment Loss : {:.4f}'.
          format(epoch_idx + 1,
                 np.mean(recon_losses),
                 np.mean(codebook_losses),
                 np.mean(commitment_losses)))
    return np.mean(losses)

def train(config, data_loader, save_option=True): # learning_rate, sample=None
    ######## Set the desired seed value #######
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # print(args.seed)
    #######################################

    # Create the model and dataset
    model = get_model(config).to(device)
    # if data_loader is None:
    #     if sample: data_loader = load_data(config['train_params']['path'],sample_data=sample, shuffle_data=False, batch_size=batch_size)
    #     else: data_loader = load_data(config['train_params']['path'])
    num_epochs = config['train_params']['epochs']
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True)
    criterion = {
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss()
    }.get(config['train_params']['crit'])

    # Create output directories
    if not os.path.exists(config['train_params']['output_dir']):
        os.mkdir(config['train_params']['output_dir'])

    # Train the model
    best_loss = np.inf

    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch(epoch_idx, model, data_loader, optimizer, criterion, config)
        # scheduler.step(mean_loss)
        # Simply update checkpoint if found better version
        if mean_loss < best_loss:
            print('Improved Loss to {:.4f} .... Saving Model'.format(mean_loss))
            best_loss = mean_loss
        else:
            print('No Loss Improvement')

    # Save the final model and codebook indices
    if save_option:
        all_indices = []
        with torch.no_grad():
            for im, label in tqdm(data_loader, desc='Recall trained model to save codebook indices', leave=False): # Ignore the label in DataLoader
                im = im.float().to(device)
                model_output = model(im)
                indices = model_output['quantized_indices']
                all_indices.append(indices.cpu())

        # Concatenate all indices into a single tensor and save it
        indices_tensor = torch.cat(all_indices, dim=0)
        torch.save(indices_tensor, os.path.join(config['train_params']['output_dir'], config['train_params']['indices_tensor']))

        torch.save(model.state_dict(), os.path.join(config['train_params']['output_dir'], config['train_params']['model_name']))


def validate(config, data_loader): #sample, batch_size
    ######## Load saved model and assign it to a similar structure ########
    state_dict = torch.load(os.path.join(config['train_params']['output_dir'], config['train_params']['model_name']))
    model = get_model(config).to(device)
    model.load_state_dict(state_dict)
    #######################################

    # Validate the model with validation set
    # val_loader = load_data(config['train_params']['path'],sample_data=sample, shuffle_data=False, batch_size=batch_size)
    criterion = {
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss()
    }.get(config['train_params']['crit'])

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        total_recon_loss = 0
        total_codebook_loss = 0
        total_commitment_loss = 0
        total_loss = 0
        for im, label in data_loader:
            im = im.float().to(device)
            model_output = model(im)
            output = model_output['generated_image']

            recon_loss = criterion(output, im)
            quantize_losses = model_output['quantized_losses']

            total_recon_loss += recon_loss.item()
            total_codebook_loss += quantize_losses['codebook_loss'].item()
            total_commitment_loss += quantize_losses['commitment_loss'].item()
            total_loss += (config['train_params']['reconstruction_loss_weight']*recon_loss.item() +
                    config['train_params']['codebook_loss_weight']*quantize_losses['codebook_loss'].item() +
                    config['train_params']['commitment_loss_weight']*quantize_losses['commitment_loss'].item())


        avg_recon_loss = np.mean(total_recon_loss) # / len(val_loader)
        avg_codebook_loss = np.mean(total_codebook_loss) # / len(val_loader)
        avg_commitment_loss = np.mean(total_commitment_loss) # / len(val_loader)
        avg_loss = np.mean(total_loss) # / len(val_loader)
        print('Total Validation Loss : {:.4f} | Recon Loss : {:.4f} | Codebook Loss : {:.4f} | Commitment Loss : {:.4f}'.
              format(avg_loss, avg_recon_loss, avg_codebook_loss, avg_commitment_loss))

        return avg_loss

def assess_test(config, sample):
    ######## Load saved model and assign it to a similar structure ########
    state_dict = torch.load(os.path.join(config['train_params']['output_dir'], config['train_params']['model_name']))
    model = get_model(config).to(device)
    model.load_state_dict(state_dict)
    #######################################

    # Validate the model with validation set
    test_loader = load_data(config['train_params']['path'],sample_data=sample, shuffle_data=False)
    criterion = {
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss()
    }.get(config['train_params']['crit'])

    # Set model to evaluation mode
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
            total_codebook_loss += quantize_losses['codebook_loss'].item()
            total_commitment_loss += quantize_losses['commitment_loss'].item()
            total_loss += (config['train_params']['reconstruction_loss_weight']*recon_loss.item() +
                    config['train_params']['codebook_loss_weight']*quantize_losses['codebook_loss'].item() +
                    config['train_params']['commitment_loss_weight']*quantize_losses['commitment_loss'].item())


        avg_recon_loss = np.mean(total_recon_loss) # / len(test_loader)
        avg_codebook_loss = np.mean(total_codebook_loss) # / len(test_loader)
        avg_commitment_loss = np.mean(total_commitment_loss) # / len(test_loader)
        avg_loss = np.mean(total_loss) # / len(test_loader)
        print('Total Test Loss : {:.4f} | Recon Loss : {:.4f} | Codebook Loss : {:.4f} | Commitment Loss : {:.4f}'.
              format(avg_loss, avg_recon_loss, avg_codebook_loss, avg_commitment_loss))

    return avg_loss


def reconstruction(config, test_data): # I have to address test_data
    idxs = torch.randint(0, len(test_dataset), (100, )) # Randomly sample indices for reconstruction.
    ims = torch.cat([test_dataset[idx][0][None, :] for idx in idxs]).float() # Create a batch of images from the test set using sampled indices.
    ims = ims.to(device)

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
    #img.save('/content/drive/My Drive/INI - Generative Episodic Memory/reconstruction_{model_name}.png'.format(model_name=config['train_params']['model_name']))
    img.save('reconstruction_{model_name}.png'.format(model_name=config['train_params']['model_name']))
    
    #plot the rexonstructions in a large figure
    plt.figure(figsize=(20, 10))
    plt.imshow(img)

    # embedding_weights = model.quantizer.embedding.weight.detach().cpu()
    #torch.save(embedding_weights, 'learned_codebook.pt')

# Cell
 ######## Read the config file #######
config_path='hyperparameters.yaml'
with open(config_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
# print(config)

def split_dataset(dataset, test_size=0.2):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)

    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=config['train_params']['seed'], shuffle=True)

    # Create Subsets for the training and testing sets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=config['train_params']['seed'], shuffle=True)

def objective(trial):
    # Define hyperparameters to tune
    # learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    # config['train_params']['lr'] = learning_rate
    batch_size = 256 #trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    num_embeddings = trial.suggest_int('num_embeddings', 20, 256)
    config['model_params']['codebook_size'] = num_embeddings
    # commitment_cost = trial.suggest_uniform('commitment_cost', 0.1, 2.0)
    # config['train_params']['commitment_loss_weight'] = commitment_cost

    # Load the dataset and split into train and test
    dataset = Shapes3DDataset(path=config['train_params']['path'])
    train_dataset, test_dataset = split_dataset(dataset)

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=config['train_params']['seed'])

    # train_losses = []
    val_losses = []

    # KFold Cross-Validation on the training set
    for train_idx, val_idx in kf.split(np.arange(len(train_dataset))): #for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_dataset))))# The code between two hash can be good if we want to do some logging on each fold index
        # Create samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # Create DataLoaders for the current fold
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)

        # train and save model while returning loss_train
        train(config, data_loader=train_loader) #sample=train_sampler, batch_size=batch_size
        # train_losses.append(train_loss)

        # validate model
        val_loss = validate(config, data_loader=val_loader) #sample=val_sampler, batch_size=batch_size
        val_losses.append(val_loss)

    # Calculate the average validation loss across folds
    avg_val_loss = np.mean(val_losses)

    return avg_val_loss


def main():
    try:
        # Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        # Best hyperparameters
        print(f"Best hyperparameters: {study.best_params}")

        # After Optuna optimization is complete, save best hyperparameters
        with open('best_hyperparameters.txt', 'w') as f:
            f.write(f"Best hyperparameters: {study.best_params}\n")

    except Exception as e:
        # Log any errors
        logging.error("An error occurred during the execution.")
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
