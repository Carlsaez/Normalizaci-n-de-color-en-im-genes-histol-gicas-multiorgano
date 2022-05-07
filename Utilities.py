import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as fn

class losses():
    def __init__(self, Adv_Loss, Cyc_Loss, Id_Loss, Disc_Loss):
        self.Adv_L = Adv_Loss
        self.Cyc_L = Cyc_Loss
        self.Id_L = Id_Loss
        self.Disc_L = Disc_Loss

    def Adv_loss(self, D_fake):
        loss = self.Adv_L(D_fake, torch.ones_like(D_fake))
        return loss

    def Cycle_loss(self, real, fake):
        loss = self.Cyc_L(real, fake)
        return loss

    def Identity_loss(self, real, identity):
        loss = self.Id_L(real, identity)
        return loss

    def Disc_loss(self, D_real, D_fake):
        D_real_loss = self.Disc_L(D_real, torch.ones_like(D_real))
        D_fake_loss = self.Disc_L(D_fake, torch.zeros_like(D_fake))
        return D_real_loss + D_fake_loss

def save_model(model, opt, path):
    model_params = model.state_dict()
    opt_params = opt.state_dict()
    chkpt = {'model': model_params, 'optimizer': opt_params}
    torch.save(chkpt, path)
    # torch.save(model_params, path_model)
    # torch.save(opt_params, path_opt)

def load_model(model, opt, path, device):
    chkpt = torch.load(path, map_location=device)
    # model_params = torch.load(path_model)
    # opt_params = torch.load(path_opt)
    model.load_state_dict(chkpt['model'])
    opt.load_state_dict(chkpt['optimizer'])
    return model, opt

def Zero_Threshold(x, t=0.1):
	# zeros = torch.cuda.FloatTensor(x.shape).fill_(0.0)
	return torch.where(x > t, x, torch.zeros_like(x))

def plot_real_fake(n_samples, loader, gen):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, ax = plt.subplots(nrows=n_samples, ncols=2, figsize=(10,30))
    # _, ax = plt.subplots(nrows=n_samples, ncols=2)

    for i, img in enumerate(loader):
        img_t = img.to(device)
        # img_t = (img_t+1)/2
        ax[i,0].imshow(fn.to_pil_image(img_t[0]))
        ax[i,0].set_title('Real Image')
        ax[i,0].grid(False)

        # img_t = img_t.to(device)
        A_fake = gen(img_t)
        ax[i,1].imshow(fn.to_pil_image(A_fake[0]))
        ax[i,1].set_title('Fake Image')
        ax[i,1].grid(False)
        if i == n_samples-1:
            break

    plt.show()

def plot_loss(metrics_train, metrics_val, all=False):
    n_epochs = len(metrics_train)
    G_loss_train = []
    D_loss_train = []
    G_loss_val = []
    D_loss_val = []
    G_adv_train = []
    G_adv_val = []
    G_cycle_train = []
    G_cycle_val = []
    G_id_train = []
    G_id_val = []

    for idx in range(len(metrics_train)):
        G_loss_train.append(metrics_train[idx]['G_loss'])
        G_loss_val.append(metrics_val[idx]['G_loss'])
        D_loss_train.append(metrics_train[idx]['D_loss'])
        D_loss_val.append(metrics_val[idx]['D_loss'])
        G_adv_train.append(metrics_train[idx]['G_adv'])
        G_adv_val.append(metrics_val[idx]['G_adv'])
        G_cycle_train.append(metrics_train[idx]['G_cycle'])
        G_cycle_val.append(metrics_val[idx]['G_cycle'])
        G_id_train.append(metrics_train[idx]['G_id'])
        G_id_val.append(metrics_val[idx]['G_id'])

    # Gráficas
    plt.style.use("ggplot")
    plt.figure(figsize=(10,10), dpi=80)
    plt.plot(np.arange(0, n_epochs), G_loss_train, label="G_train_loss")
    plt.plot(np.arange(0, n_epochs), G_loss_val, label="G_val_loss")
    plt.plot(np.arange(0, n_epochs), D_loss_train, label="D_train_loss")
    plt.plot(np.arange(0, n_epochs), D_loss_val, label="D_val_loss")
    if all:
        plt.plot(np.arange(0, n_epochs), G_adv_train, label="G_adv_train")
        plt.plot(np.arange(0, n_epochs), G_adv_val, label="G_adv_val")
        plt.plot(np.arange(0, n_epochs), G_cycle_train, label="G_cycle_train")
        plt.plot(np.arange(0, n_epochs), G_cycle_val, label="G_cycle_val")
        plt.plot(np.arange(0, n_epochs), G_id_train, label="G_id_train")
        plt.plot(np.arange(0, n_epochs), G_id_val, label="G_id_val")

    plt.title("Training Loss vs Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



def plot_loss_attention(metrics_train, metrics_val):
    n_epochs = len(metrics_train)
    LossCycleA = []
    LossCycleB = []
    AdvLossA = []
    AdvLossB = []
    DisLossA = []
    DisLossB = []

    LossCycleA_v = []
    LossCycleB_v = []
    AdvLossA_v = []
    AdvLossB_v = []
    DisLossA_v = []
    DisLossB_v = []

    for idx in range(len(metrics_train)):
        LossCycleA.append(metrics_train[idx]['LossCycleA'])
        LossCycleB.append(metrics_train[idx]['LossCycleB'])
        AdvLossA.append(metrics_train[idx]['AdvLossA'])
        AdvLossB.append(metrics_train[idx]['AdvLossB'])
        DisLossA.append(metrics_train[idx]['DisLossA'])
        DisLossB.append(metrics_train[idx]['DisLossB'])

        LossCycleA_v.append(metrics_val[idx]['LossCycleA'])
        LossCycleB_v.append(metrics_val[idx]['LossCycleB'])
        AdvLossA_v.append(metrics_val[idx]['AdvLossA'])
        AdvLossB_v.append(metrics_val[idx]['AdvLossB'])
        DisLossA_v.append(metrics_val[idx]['DisLossA'])
        DisLossB_v.append(metrics_val[idx]['DisLossB'])

    # Gráficas
    plt.style.use("ggplot")
    plt.figure(figsize=(10,10), dpi=80)
    plt.plot(np.arange(0, n_epochs), LossCycleA, label="LossCycleA")
    plt.plot(np.arange(0, n_epochs), LossCycleB, label="LossCycleB")
    plt.plot(np.arange(0, n_epochs), AdvLossA, label="AdvLossA")
    plt.plot(np.arange(0, n_epochs), AdvLossB, label="AdvLossB")
    plt.plot(np.arange(0, n_epochs), DisLossA, label="DisLossA")
    plt.plot(np.arange(0, n_epochs), DisLossB, label="DisLossB")

    plt.plot(np.arange(0, n_epochs), AdvLossA_v, label="AdvLossA_v")
    plt.plot(np.arange(0, n_epochs), AdvLossB_v, label="AdvLossB_v")
    plt.plot(np.arange(0, n_epochs), AdvLossA_v, label="AdvLossA_v")
    plt.plot(np.arange(0, n_epochs), AdvLossB_v, label="AdvLossB_v")
    plt.plot(np.arange(0, n_epochs), DisLossA_v, label="DisLossA_v")
    plt.plot(np.arange(0, n_epochs), DisLossB_v, label="DisLossB_v")

    plt.title("Training Loss vs Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()