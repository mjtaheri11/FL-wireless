import argparse
import random
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torchvision.datasets
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from munkres import Munkres

from config import config


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def packet_error_calculator(distance_matrix, channel_interference, user_max_power):
    # bandwidth and m was a little tricky. so we just used some certain integers
    q_packet_error = 1 - np.exp(-1.08 * np.divide(channel_interference + 1e-14, user_max_power * np.power(distance_matrix , (-2))))
    return q_packet_error


def sinr_calculator(distance_matrix, channel_interference_matrix, power=1):
    SINR = power * np.divide((np.power(distance_matrix, (-2))) , channel_interference_matrix)
    return SINR


def per_user_total_energy_calculator(fl_model_data_size, uplink_delay, psi, omega_i, theta, user_power):
    total_energy = psi * omega_i * (theta ** 2) * fl_model_data_size + user_power * uplink_delay 
    return total_energy


def channel_rate_calculator(bandwidth, sinr):
    rate = bandwidth * np.log2(1 + sinr)
    return rate
    

def train(args):
    
    # hungarian algorithm object
    mnkr = Munkres()
    # number of training datasamples for each device. 
    datanumber = [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100, 200, 300, 400, 500, 100]  
    # Interference over downlink 
    channel_interference_downlink = 0.06 * 0.000003        
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    if args.user_blocks == 3:
        channel_interference = (np.array([.05, 0.1, 0.14]) - 0.04) * 0.000001  # Interference over each RB
    elif args.user_blocks == 6:
        channel_interference = (np.array([ 0.05, 0.07,  0.09,  0.11,  0.13,  0.15])-0.04)*0.000001  # Interference over each RB
    elif args.user_blocks == 9:
        channel_interference = (np.array([0.03, 0.06, 0.07, 0.08,  0.1, 0.11, 0.12, 0.14, 0.15])-0.04)*0.000001  # Interference over each RB

    num_resource_blocks = len(channel_interference)
    # writer = SummaryWriter()

    for average in range(1, args.averagenumber + 1):
        # The distance between the users and the BS
        distance = np.random.rand(args.user_number, 1) * 500           
        # Packet error rate of each user over each RB
        q_packet_error = packet_error_calculator(distance, channel_interference, config['channel_parameters']['user_max_power'])           
        # SINR of each user over each RB
        sinr = sinr_calculator(distance, channel_interference, config["channel_parameters"]["user_max_power"])  
        # Uplink data rate of each user over each RB
        rateu = channel_rate_calculator(
            config['channel_parameters']["uplink_bandwidth"],
            sinr 
            )

        # downlink SINR of each user
        SINRd = sinr_calculator(distance, channel_interference_downlink)        
        # downlink data rate of each user
        rated = channel_rate_calculator(                                         
            config['channel_parameters']['downlink_bandwidth'],
            SINRd
            )

        models = {}
        for user in range(args.user_number):
            network_user = f"model_{user}"

            # Create the model
            model = NeuralNet(
                config["model_hyperparameters"]["input_size"],
                config["model_hyperparameters"]["hidden_size"],
                config["model_hyperparameters"]["num_classes"]
                ).to(device)
            
            models[network_user] = model
                                
        # Data size of each FL model, we assume that each element occupies 16 bits 
        total_model_params = sum(p.numel() for p in models['model_0'].parameters())
        Z = total_model_params * 16 / 1024 / 1024                                   
        # Uplink delay of each user over each RB 
        delayu = np.divide(Z, rateu)                                   
        # Downlink delay of each user
        delayd = np.divide(Z, rated)                                   
        # Sum downlink delay of each user
        totaldelay = delayu + delayd                                   

        # Sum energy consumption of each user
        totalenergy = per_user_total_energy_calculator(Z,
                                                        delayu,
                                                        config['channel_parameters']['psi'],
                                                        config['channel_parameters']['omega_i'],
                                                        config['channel_parameters']['theta'],
                                                        config["channel_parameters"]["user_max_power"],
                                                        )
        
        # ============ proposed algorithm ============ 
        # edge matrix for Hungarian algorithm 
        W = np.zeros((args.user_number, num_resource_blocks)) 
        if args.strategy == 'proposed' or args.strategy == 'baseline1':

            # Set value for each adge according to the equation (24)
            for i in range(args.user_number):
                for j in range(num_resource_blocks):
                    if (totaldelay[i, j] < config['channel_parameters']['delay_requirement']) and (totalenergy[i, j] < config['channel_parameters']['energy_requirement']):
                        W[i, j] = datanumber[i] * (q_packet_error[i, j])
                    else:
                        W[i, j] = 1e+10  # here we use a very large number instead of zero for doing our hungarian algorithm
            
            # Use Hungarian algorithm to find the optimal RB allocation
            assignment = mnkr.compute(W.tolist())

            if args.strategy == 'proposed':
                finalq = np.ones((1, args.user_number))
                for i, j in assignment:
                    if W[i, j] != 1e+10:
                        finalq[0, i] = q_packet_error[i, j]

            elif args.strategy == 'baseline1':
                qassignment = np.zeros(1, args.user_number)

                if num_resource_blocks < args.user_number:
                    # qassignment[0, ]
                    pass
                
        elif args.strategy == 'baseline1':
            pass 
        
    
    users_per_iteration = defaultdict(list)
    error = np.zeros((args.iteration, 1))
    iterationtime = np.zeros((args.iteration, 1))
    
    
    if len(np.where(finalq < 1)[0]) > 0:
        for iteration in range(args.iteration):
            for user in range(args.user_number):
                if (iteration == 0 and finalq[0, user] != 1) or random.random() > finalq[0, user]:

                    #  set the users in each iteration
                    users_per_iteration[iteration].append(user)                 
                    # Set input data
                    user_train_dataset = [(train_dataset[i][0], train_dataset[i][1]) for i in range(sum(datanumber[:user]), sum(datanumber[:user+1]))]     
                    train_loader = DataLoader(dataset=user_train_dataset, batch_size=args.batch_size, shuffle=True)
                    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
                    
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

                    # Training loop
                    total_step = len(train_loader)
                    for i, (images, labels) in enumerate(train_loader):
                        images = images.reshape(-1, 28*28).to(device)
                        labels = labels.to(device)

                        # Forward pass
                        outputs = models[f"model_{user}"](images)
                        loss = criterion(outputs, labels)

                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if (i+1) % 50 == 0:
                            print(f'model for user {user} Step [{i+1}/{total_step}], Loss: {loss.item()}')

            # ============ calculate the global FL model ============
            # calculate the number of users join the FL iteration i
            if len(users_per_iteration[iteration]) > 0:
                global_weights = models[f"model_0"].state_dict()
                for j, _user in enumerate(users_per_iteration[iteration]):
                    if iteration == 0 and j == 0:  
                        for layer_name in global_weights.keys():
                            global_weights[layer_name] = models[f"model_{_user}"].state_dict()[layer_name] * datanumber[_user]
                    else:
                        for layer_name in global_weights.keys():
                            global_weights[layer_name] += models[f"model_{_user}"].state_dict()[layer_name] * datanumber[_user]
                
                # divide the total number of data samples 
                for layer_name in models["model_0"].state_dict():
                    global_weights[layer_name] /= sum([datanumber[val] for val in users_per_iteration[iteration]])                   
                            
                # Create the global model
                global_model = NeuralNet(
                    config["model_hyperparameters"]["input_size"],
                    config["model_hyperparameters"]["hidden_size"],
                    config["model_hyperparameters"]["num_classes"]
                    ).to(device)

                # load the weights
                global_model.load_state_dict(global_weights)
                                    
                # Evaluation
                global_model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images = images.reshape(-1, 28*28).to(device)
                        labels = labels.to(device)
                        outputs = global_model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    test_accuracy = 100 * correct / total
                    print(f'Test Accuracy of the model on the 10000 test images: {test_accuracy}%')                    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration",
        type=float,
        default=130,      
        help="number of total iterations",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size of the model",
        )
    
    parser.add_argument(
        "--averagenumber",
        type=int,
        default=1,
        help="Number of implementation of FL",
        )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
        )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default='proposed',
        help="the strategy of FL training",
        choices=['proposed', 'baseline1', 'baseline2', 'baseline3']
        )
    
    parser.add_argument(
        "--user_number",
        type=int,
        default=9,
        help="Total number of users that implement FL",
        )
    
    parser.add_argument(
        "--user_blocks",
        type=int,
        default=6,
        choices=[3, 6, 9],
        help="Total number of user blocks that implement FL",
        )
    
    args = parser.parse_args()
    train(args)
    
