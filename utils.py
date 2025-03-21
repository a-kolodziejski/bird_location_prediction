"""
    All helper functions are gathered in this file
"""
    
# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import torch


    
def load_and_preprocess_df(df_filepath, max_speed, q = None, sampling_interval = None):
    '''
    Loads the data, performs basic validation, removes super-sonic birds and (optionally) removes uuids with big number of observations (they probably do not correspond to single bird observations) and (optionally) downsamples the data, i.e. if for example sampling_interval = '1s' it calculates mean position within 1 second interval.
    
    Args:
        df_filepath (str): filepath of DataFrame to preprocess
        max_speed (float): maximum bird speed allowed. All rows of DataFrame in which this value is exceeded will be removed
        q (float): quantile level, if > 0 - removes uuids with extreme number of observation
        sampling_interval (string): time interval over which data is averaged
    Returns:
        df (pandas.DataFrame): Loaded and preprocessed dataframe
    '''
    # LOAD DATA AND PERFORM BASIC VALIDATION
    df = pd.read_csv(df_filepath, 
                 delimiter = ';', 
                 parse_dates = ['time'], 
                usecols = ['uuid', 'x', 'y', 'z', 'time'])
    # Calculate number of rows
    print(f"There are {len(df)} rows in the DataFrame")
    # Check for missing values
    print(f"There are {df.isna().sum().sum()} rows with missing values in the DataFrame")
    # OUTLIER REMOVAL
    if q:
        # Calculate max number of observations
        max_uuid = df['uuid'].value_counts().quantile(q)
        # Get rid of observations of more than max_uuid points
        df['count'] = df.groupby('uuid')['uuid'].transform('count')
        df = df[df['count'] < max_uuid].copy()
        print(f"There are {len(df)} rows after removing uuids with too many observations")
        # HANDLE DOWNSAMPLING
    if sampling_interval:
        df = df.groupby('uuid').resample(sampling_interval, on = 'time').mean().copy()
        df.dropna(inplace = True)
        df.reset_index(inplace = True)
        print(f"There are {len(df)} rows after downsampling")
    # HANDLE SPEED OF BIRDS
    # Normalize time within each flight
    df['time'] = df.groupby('uuid')['time'].transform(lambda x: x - x.min())
    # Convert to seconds
    df['time'] = df['time'].dt.total_seconds()
    # Calculate position increments
    delta_x = df.groupby('uuid')['x'].transform(lambda x: x - x.shift(1))
    delta_y = df.groupby('uuid')['y'].transform(lambda x: x - x.shift(1))
    delta_z = df.groupby('uuid')['z'].transform(lambda x: x - x.shift(1))
    # Calculate distance travelled
    distance = (delta_x**2 + delta_y**2 + delta_z**2)**(1/2)
    # Calculate time interval between consecutive observations
    df['delta_t'] = df['time'] - df['time'].shift(1)
    # Calculate speed
    df['speed'] = distance/df['delta_t']
    df['speed'] = df['speed'].shift(-1)
    df['speed'].fillna(0.0, inplace = True)
    # Get only ordinary birds
    df_to_drop = df[df['speed'] > max_speed].copy()
    # See what uuids are in df_to_drop
    bad_uuids = set(df_to_drop['uuid'])
    # Get the list of all uuids
    all_uuids = set(df['uuid'])
    # Get the set difference
    good_uuids = all_uuids - bad_uuids
    # Select only rows with good uuids
    df = df[df['uuid'].isin(good_uuids)].copy()
    print(f"There are {len(df)} rows after removing super-sonic birds")
    print('DONE')
    return df    


def windowify(df, window_size, horizon, cols = ['time', 'x', 'y', 'z']):
    '''
    Splits timeseries data into windows of given size. 
    
    Args:
        df (dataframe): bird movement data
        window_size (int): size of window
        horizon (int): number of timesteps into the future to predict
        cols (list of strings): list of column names to windowify
    Returns:
        X (list of lists of floats): list of windows of size window_size
        y (list of lists of floats): values to predict (windows of horizon size)
    '''
    X = []
    y = []
    # Get all uuids
    uuid_list = df['uuid'].unique()
    # Loop over uuids
    for uuid in tqdm(uuid_list):
        # Calculate num of observations in sample
        num_of_observations = len(df[df['uuid'] == uuid])
        # Check if window size is smaller than number of observations in uuid
        if num_of_observations < window_size + horizon:
            continue
        else:
        # Select appropriate observations
            observations = df[df['uuid'] == uuid][cols].copy()
            for i in range(num_of_observations - window_size - horizon):
                # From observations pick sample of window_size + horizon (horizon will be prediction)
                sample = observations.iloc[i:i+window_size+horizon].copy()
                # Normalize time within each sample
                sample.loc[:,'time'] = sample.loc[:,'time'] - sample.loc[:,'time'].iloc[0]
                # Consecutive points must differ by 1 second
                if sample['time'].iloc[-1] > window_size + horizon - 1:
                    continue
                # Add sequence of row vectors od WINDOW_SIZE to X
                X.append(sample.iloc[0:window_size].values)
                # Add prediction vector to y
                y.append(sample.iloc[window_size:].values)
    print(f"There are {len(X)} windows")            
    print('DONE')
    return X, y

def save_windows(X, y, X_filepath, y_filepath):
    '''
    Saves created windows at specified locations.
    
    Args:
        X (list of lists of floats): list of windows to save
        y (list of lists of floats): list of predictions to save
        X_filepath (str): filepath for storing X
        y_filepath (str): filepath for storing y
    '''
    # Get current working directory
    cur_dir = os.getcwd()
    # Define save directory
    save_dir = cur_dir + "\save_data"
    # If folder does not exist then create it
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Save X
    with open(save_dir + X_filepath, 'wb') as f:
        print(f'Saving X to {save_dir + X_filepath}')
        pickle.dump(X, f)

    # Save y
    with open(save_dir + y_filepath, 'wb') as f:
        print(f'Saving y to {save_dir + y_filepath}')
        pickle.dump(y, f)
    print("DONE")
        

def load_windows(X_filepath, y_filepath):
    '''
    Loads saved windows.
    
    Args:
        X_name (str): filename for X
        y_name (str): filename for y
    Returns:
        X (list of lists of floats): list of windows to load
        y (list of lists of floats): list of predictions to load
    '''
    # Get current working directory
    cur_dir = os.getcwd()
    # Define save directory
    save_dir = cur_dir + "\save_data"
    # Load X
    with open(save_dir + X_filepath, 'rb') as f:
        X = pickle.load(f)

    # Load y
    with open(save_dir + y_filepath, 'rb') as f:
        y = pickle.load(f)
    print("DONE")   
    # Return X and y
    return X, y


def prepare_data(X, y, train_proportion = 0.8, val_proportion = 0.1, keeptime = False):
    '''
    Prepares data for training i.e. turns windows into tensors, shuffles them
    and divides into train, val and test sets.
    
    Args:
        X (list of lists of floats): list of windows
        y (list of lists of floats): list of prediction windows
        train_proportion (float): percentage of training data
        val_proportion (float): percentage of validation data
        keeptime (bool): if True, time is also added to train, val and test data
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (torch.tensor): windowed dataset split into train, val and test tensors
    '''
    # Turn into tensors
    X = torch.tensor(X.copy(), dtype = torch.float32)
    y = torch.tensor(y.copy(), dtype = torch.float32)
    
    if not keeptime:
        # Keep only coordinates (not time)
        X = X[:, :, 1:]
        y = y[:, :, 1:]

    # Shuffle indices
    shuffled_indices = np.random.permutation(len(X))

    # Shuffle X and y
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    # Define train, val and test sets
    num_of_training_examples = int(train_proportion * len(X))
    num_of_val_examples = int(val_proportion * len(X))

    # Tarin set
    X_train = X[:num_of_training_examples]
    y_train = y[:num_of_training_examples]

    # Validation set
    X_val = X[num_of_training_examples:num_of_training_examples + num_of_val_examples]
    y_val = y[num_of_training_examples:num_of_training_examples + num_of_val_examples]

    # Test set
    X_test = X[num_of_training_examples + num_of_val_examples:]
    y_test = y[num_of_training_examples + num_of_val_examples:]
    
    print(f"Number of training examples: {len(X_train)}, validation examples: {len(X_val)} and test examples: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train(model, X_train, y_train, batch_size, num_epochs, optimizer, loss_fn, X_val, y_val, writer):
    '''
    Implements training loop for a given model.
    
    Args:
        model (torch.nn.Module): model to train and avaluate
        X_train (torch.tensor): training data
        y_train (torch.tensor): training labels
        batch_size (int): size of a single batch of training examples
        num_epochs (int): number of training iterations
        optimizer (torch.optim): optimizer used for training
        loss_fn (torch.nn.functional): loss function
        X_val (torch.tensor): validation data to assess the model performance during training
        y_val (torch.tensor): validation data labels
        writer (torch.utils.tensorboard): to help keep track of training
    '''
    # Calculate number of training samples
    num_of_training_samples = len(X_train)

    for epoch in tqdm(range(num_epochs)):
        # Draw batch of samples from train set
        batch_indices = torch.randint(low = 0, high = num_of_training_samples, size = (batch_size,))
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        # Activate training mode in model
        model.train()
        # Pass X_batch through the model
        y_pred = model(X_batch)
        # Calculate loss (square root of mean squared error)
        loss = torch.sqrt(loss_fn(y_pred, y_batch))
        # Add loss to writer
        writer.add_scalar("Train loss", loss, epoch)
        # Reset optimizer grads
        optimizer.zero_grad()
        # Run backpropagation
        loss.backward()
        # Update model's parameters
        optimizer.step()
        # Once in a while print training loss
        # Switch to eval mode
        model.eval()
        with torch.inference_mode():
            # Pass X_val through the model and calculate loss
            y_val_pred = model(X_val)
            val_loss = torch.sqrt(loss_fn(y_val_pred, y_val))
            writer.add_scalar("Val loss", loss, epoch)
        if epoch % 1000 == 0:
            print(f"Training loss at epoch {epoch}: {loss.item()} and val loss: {val_loss.item()}")
    # Close writer object
    writer.close()


def naive_forecast(X, horizon):
    '''
    Implements naive forecast which is going to be the baseline to beat by NN approach.
    Naive forecast calculates velocity out of last two known bird positions and predicts next
    position by applying simple kinematics equations of motion.
    
    Args:
        X (torch.tensor of dimension (batch_size, window_size, num_features)): batch of windows to predict next position
        horizon (int): number of timesteps into the future to predict
    Returns:
        y (torch.tensor of dimension (batch_size, num_features)): batch of next naive predictions
    '''
    # Extract next to last positions from bacth of windows
    next_to_last_pos = X[:, -2, : ]
    # Extract last positions from bacth of windows
    last_pos =  X[:, -1, : ]
    # Calculate velocity
    velocity =  last_pos - next_to_last_pos
    # Initialize predictions tensor
    y = torch.zeros(size = (X.shape[0], horizon, X.shape[2]))
    # Calculate next positions
    for t in range(1, horizon + 1):
        y[:, t - 1, :] = last_pos + t*velocity
    return y