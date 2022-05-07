import torch
import torch.nn as nn
from Utilities import losses


def train(gen_A, gen_B, disc_A, disc_B, opt_gen, opt_disc_A, opt_disc_B, loader_A, loader_B, alpha, lamb, gamma, eval=False):
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()

    loss = losses(Adv_Loss=bce, Cyc_Loss=l1, Id_Loss=l1, Disc_Loss=mse)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    for idx, images in enumerate(zip(loader_A, loader_B)):
        A = images[0]
        B = images[1]

        A = A.to(device)
        B = B.to(device)
        
        # Train Discriminator_A and Discriminator_B
        with torch.no_grad() if eval else torch.cuda.amp.autocast():
          A_fake = gen_A(B)
          D_A_real = disc_A(A)
          D_A_fake = disc_A(A_fake.detach())
          D_A_loss = loss.Disc_loss(D_A_real, D_A_fake)
          
          B_fake = gen_B(A)
          D_B_real = disc_B(B)
          D_B_fake = disc_B(B_fake.detach())
          D_B_loss = loss.Disc_loss(D_B_real, D_B_fake)

          D_loss = (D_A_loss + D_B_loss)/2

        if not eval:
          opt_disc_A.zero_grad()
          D_A_loss.backward()
          opt_disc_A.step()

          opt_disc_B.zero_grad()
          D_B_loss.backward()
          opt_disc_B.step()

        # Train Generator_A and Generatos_B

        with torch.no_grad() if eval else torch.cuda.amp.autocast():
          D_A_fake = disc_A(A_fake)
          D_B_fake = disc_B(B_fake)
                  
          # Adversarial loss
          G_A_adv_loss = loss.Adv_loss(D_A_fake)
          G_B_adv_loss = loss.Adv_loss(D_B_fake)
          G_adv_loss = G_A_adv_loss + G_B_adv_loss

          # Cycle loss
          A_cycle = gen_A(B_fake)
          G_A_cycle_loss = loss.Cycle_loss(A, A_cycle)

          B_cycle = gen_B(A_fake)
          G_B_cycle_loss = loss.Cycle_loss(B, B_cycle)

          G_cycle_loss = G_A_cycle_loss + G_B_cycle_loss

          # Identity loss
          A_identity = gen_A(A)
          G_A_id_loss = loss.Identity_loss(A, A_identity)

          B_identity = gen_B(B)
          G_B_id_loss = loss.Identity_loss(B, B_identity)

          G_id_loss = G_A_id_loss + G_B_id_loss

          # Generator loss
          G_loss = (alpha*(G_adv_loss) + lamb*(G_cycle_loss) + gamma*(G_id_loss))

        if not eval:
          opt_gen.zero_grad()
          G_loss.backward()
          opt_gen.step()

    metrics = {'G_adv':G_adv_loss.item(), 'G_cycle':G_cycle_loss.item(), 'G_id':G_id_loss.item(), 'G_loss':G_loss.item(), 'D_loss':D_loss.item()} # 
    return metrics