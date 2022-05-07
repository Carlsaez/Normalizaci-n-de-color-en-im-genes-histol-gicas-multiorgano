import torch
import Utilities
from torch.optim import Adam
import Resnet_UPSampling
import ResUnet
import Unet_UPsampling
import itertools
from torch.utils.data import DataLoader
from load_dataset import ImageDataset
import numpy as np
import os
from Train import train

def main(path_A, path_B, path_model, model, number_res, n_epochs, batch_size, l_rate, alpha, lamb, gamma, resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate model
    if model == 'Resnet':
        gen_A = Resnet_UPSampling.Generator(input_c=3, number_res=number_res).to(device)
        gen_B = Resnet_UPSampling.Generator(input_c=3, number_res=number_res).to(device)
        disc_A = Resnet_UPSampling.Discriminator(inputs=3).to(device)
        disc_B = Resnet_UPSampling.Discriminator(inputs=3).to(device)
    
    elif model == 'ResUnet':
        gen_A = ResUnet.Generator(input_c=3).to(device)
        gen_B = ResUnet.Generator(input_c=3).to(device)
        disc_A = ResUnet.Discriminator(inputs=3).to(device)
        disc_B = ResUnet.Discriminator(inputs=3).to(device)

    elif model == 'Unet':
        gen_A = Unet_UPsampling.Generator(input_c=3).to(device)
        gen_B = Unet_UPsampling.Generator(input_c=3).to(device)
        disc_A = Unet_UPsampling.Discriminator(inputs=3).to(device)
        disc_B = Unet_UPsampling.Discriminator(inputs=3).to(device)

    optimizer_G = Adam(itertools.chain(gen_A.parameters(), gen_B.parameters()), lr=l_rate, betas=(0.5, 0.999), eps=1e-08, weight_decay=0)
    optimizer_D_A = Adam(disc_A.parameters(), lr=l_rate, betas=(0.5, 0.999), eps=1e-08, weight_decay=0)
    optimizer_D_B = Adam(disc_B.parameters(), lr=l_rate, betas=(0.5, 0.999), eps=1e-08, weight_decay=0)

    metrics_t = []
    metrics_v = []

    # Load the checkpoint
    if resume:  
      Utilities.load_model(gen_A, optimizer_G, os.path.join(path_model,'gen_A.pt'), device)
      Utilities.load_model(gen_B, optimizer_G, os.path.join(path_model,'gen_B.pt'), device)
      Utilities.load_model(disc_A, optimizer_D_A, os.path.join(path_model,'disc_A.pt'), device)
      Utilities.load_model(disc_B, optimizer_D_B, os.path.join(path_model,'disc_B.pt'), device)
      metrics_t = list(np.load(os.path.join(path_model,'metrics_train.npy'), allow_pickle=True))
      metrics_v = list(np.load(os.path.join(path_model,'metrics_val.npy'), allow_pickle=True))


    data_A_train = ImageDataset(path_A, data_group='train', transform=None)
    data_B_train = ImageDataset(path_B, data_group='train', transform=None)
    loader_A_train = DataLoader(data_A_train, batch_size, shuffle=True)
    loader_B_train = DataLoader(data_B_train, batch_size, shuffle=True)

    data_A_val = ImageDataset(path_A, data_group='val', transform=None)
    data_B_val = ImageDataset(path_B, data_group='val', transform=None)
    loader_A_val = DataLoader(data_A_val, batch_size, shuffle=True)
    loader_B_val = DataLoader(data_B_val, batch_size, shuffle=True)

    
    for epoch in range(n_epochs):
        metrics_train = train(gen_A, gen_B, disc_A, disc_B, optimizer_G, optimizer_D_A, optimizer_D_B, loader_A_train, loader_B_train, alpha, lamb, gamma, eval=False)
        metrics_val = train(gen_A, gen_B, disc_A, disc_B, optimizer_G, optimizer_D_A, optimizer_D_B, loader_A_val, loader_B_val, alpha, lamb, gamma, eval=True)

        Utilities.save_model(gen_A, optimizer_G, os.path.join(path_model,'gen_A.pt'))
        Utilities.save_model(gen_B, optimizer_G, os.path.join(path_model,'gen_B.pt'))
        Utilities.save_model(disc_A, optimizer_D_A, os.path.join(path_model,'disc_A.pt'))
        Utilities.save_model(disc_B, optimizer_D_B, os.path.join(path_model,'disc_B.pt'))

        print(f"Nº epoch: {epoch}\n")
        print(f"Metrics Train:\n {metrics_train}")
        metrics_t.append(metrics_train)
        np.save(os.path.join(path_model,'metrics_train.npy'), np.array(metrics_t))

        print(f"Metrics Validation:\n {metrics_val}\n")
        metrics_v.append(metrics_val)
        np.save(os.path.join(path_model,'metrics_val.npy'), np.array(metrics_v))

    return metrics_t, metrics_v