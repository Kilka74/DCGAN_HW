from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import wandb
from piq import FID
from piq.ssim import ssim
import torch
from torch import nn
import torchvision.transforms as T
import torchvision


def train_gan(
        discriminator,
        optimizer_discriminator,
        generator,
        optimizer_generator,
        train_loader,
        noise_dim,
        num_steps,
        eval_freq,
        wandb_log=False,
        device=torch.device("cuda:0")
):
    discriminator.train()
    generator.train()
    d_losses = []
    g_losses = []
    val_d_losses = []
    val_g_losses = []
    val_fid = []
    val_ssim = []
    criterion = nn.BCEWithLogitsLoss()
    for step in tqdm(range(num_steps), desc="Train dcgan"):
        real_batch = next(train_loader).to(device)
        latent = torch.randn((real_batch.shape[0], noise_dim), device=device)
        fake_batch = generator(latent).detach()
        real_labels = torch.ones((real_batch.shape[0],), dtype=torch.float, device=device)
        fake_labels = torch.zeros((real_batch.shape[0],), dtype=torch.float, device=device)
        real_logits = discriminator(real_batch).view(-1)
        fake_logits = discriminator(fake_batch).view(-1)
        discriminator_loss = criterion(real_logits, real_labels) + criterion(fake_logits, fake_labels)

        optimizer_discriminator.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        optimizer_discriminator.step()
        d_losses.append((step, discriminator_loss.item()))

        fake_generations = discriminator(generator(latent)).view(-1)
        generator_loss = criterion(fake_generations, real_labels)

        optimizer_generator.zero_grad(set_to_none=True)
        generator_loss.backward()
        optimizer_generator.step()
        g_losses.append((step, generator_loss.item()))

        if wandb_log:
            wandb.log({
                "generator_lr": optimizer_generator.param_groups[0]["lr"],
                "discriminator_lr": optimizer_discriminator.param_groups[0]["lr"],
                "train_generator_loss": generator_loss.item(),
                "train_discriminator_loss": discriminator_loss.item()
            }, step=step)

        if step % eval_freq == 0:
            discriminator.eval()
            generator.eval()
            val_real_set = []
            val_fake_set = []
            val_discriminator_loss = 0
            val_generator_loss = 0
            fid = 0
            ssim_metric = 0
            unnormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
            with torch.no_grad():
                fid_metric = FID()
                for _ in range(10):
                    real_batch = next(train_loader).to(device)
                    latent = torch.randn((real_batch.shape[0], noise_dim), device=device)
                    fake_batch = generator(latent)
                    val_real_set.append(real_batch)
                    val_fake_set.append(fake_batch)
                    real_labels = torch.ones((real_batch.shape[0],), dtype=torch.float, device=device)
                    fake_labels = torch.zeros((real_batch.shape[0],), dtype=torch.float, device=device)
                    real_logits = discriminator(real_batch).view(-1)
                    fake_logits = discriminator(fake_batch).view(-1)
                    discriminator_loss = criterion(real_logits, real_labels) + criterion(fake_logits, fake_labels)
                    generator_loss = criterion(fake_logits, real_labels)
                    val_discriminator_loss += discriminator_loss.item() * 0.1
                    val_generator_loss += generator_loss.item() * 0.1

                val_d_losses.append((step, val_discriminator_loss))
                val_g_losses.append((step, val_generator_loss))
                
                val_real_set = unnormalize(torch.cat(val_real_set, dim=0))
                val_real_set_reshaped = val_real_set.reshape(val_real_set.shape[0], -1)
                val_fake_set = unnormalize(torch.cat(val_fake_set, dim=0))
                val_fake_set_reshaped = val_fake_set.reshape(val_fake_set.shape[0], -1)
                fid = fid_metric(val_real_set_reshaped, val_fake_set_reshaped)
                
                ssim_metric = ssim(val_real_set, val_fake_set)
                val_fid.append((step, fid.item()))
                val_ssim.append((step, ssim_metric.item()))
                # logging image to wandb
                noise = torch.randn((64, 100), device=device)
                res_pictures = unnormalize(generator(noise).detach())
                images = torch.clip(torchvision.utils.make_grid(res_pictures).permute((1, 2, 0)), 0, 1)
                
                if wandb_log:
                    wandb.log({
                        "val_generator_loss": val_generator_loss,
                        "val_discriminator_loss": val_discriminator_loss,
                        "validation FID": fid.item(),
                        "validation SSIM": ssim_metric.item(),
                        "images": wandb.Image(images.cpu().numpy())
                    })
            discriminator.train()
            generator.train()
            clear_output()
            fig, axs = plt.subplots(1, 4, figsize=(24, 6))
            axs[0].scatter(*zip(*d_losses), alpha=0.1, color='blue', label='train d loss')
            axs[0].plot(*zip(*val_d_losses), color='red', label='val d loss')
            axs[1].scatter(*zip(*g_losses), alpha=0.1, color='red', label='train g loss')
            axs[1].plot(*zip(*val_g_losses), color='red', label='val g loss')
            axs[2].plot(*zip(*val_fid), color='red', label="val fid")
            axs[3].plot(*zip(*val_ssim), color='red', label="val ssim")
            axs[0].grid()
            axs[0].legend()
            axs[1].grid()
            axs[1].legend()
            axs[2].grid()
            axs[2].legend()
            axs[3].grid()
            axs[3].legend()
            plt.tight_layout()
            plt.show()

       
def _save_checkpoint(discr, generator, optimizer_d, optimizer_g, name):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    state = {
        "discriminator_state_dict": discr.state_dict(),
        "generator_state_dict": generator.state_dict(),
        "optimizer_gen": optimizer_g.state_dict(),
        "optimizer_discr": optimizer_d.state_dict(),
    }
    filename = str("{}.pth".format(name))
    torch.save(state, filename)
