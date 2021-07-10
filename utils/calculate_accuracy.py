# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/calculate_accuracy.py


from utils.sample import sample_latents
from utils.losses import calc_derv4lo, latent_optimise

import numpy as np
from scipy import linalg
from tqdm import tqdm
import math
import torch.nn.functional as F

import torch
import os
from torch.nn import DataParallel
from torchvision.utils import save_image
import shutil



def calculate_accuracy(dataloader, generator, discriminator, D_loss, num_evaluate, truncated_factor, prior, latent_op,
                       latent_op_step, latent_op_alpha, latent_op_beta, device, consistency_reg, eval_generated_sample=False, weight_regularizer = None, save=False, checkpoint_dir=None, logger=None, reg_transform=None):
    generator.eval()
    discriminator.eval()
    data_iter = iter(dataloader)
    batch_size = dataloader.batch_size

    generated_images = []
    generated_labels = []

    if isinstance(generator, DataParallel):
        z_dim = generator.module.z_dim
        num_classes = generator.module.num_classes
        conditional_strategy = discriminator.module.conditional_strategy
    else:
        z_dim = generator.z_dim
        num_classes = generator.num_classes
        conditional_strategy = discriminator.conditional_strategy

    if num_evaluate % batch_size == 0:
        total_batch = num_evaluate//batch_size
    else:
        total_batch = num_evaluate//batch_size + 1

    if D_loss.__name__ == "loss_dcgan_dis" or D_loss.__name__ == "loss_rel_dis" :
        cutoff = 0.5
    elif D_loss.__name__ == "loss_hinge_dis":
        cutoff = 0.0
    elif D_loss.__name__ == "loss_wgan_dis":
        raise NotImplementedError

    print("Calculating Accuracies....")

    if eval_generated_sample:
        confidence_stats = np.zeros(num_classes)
        for batch_id in tqdm(range(total_batch)):
            z, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, device)
            if latent_op:
                z = latent_optimise(z, fake_labels, generator, discriminator, latent_op_step, 1.0, latent_op_alpha,
                                    latent_op_beta, False, device)
            try:
                if consistency_reg:
                    real_images, real_labels, real_images_aug = next(data_iter)
                else:
                    real_images, real_labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                if consistency_reg:
                    real_images, real_labels, real_images_aug = next(data_iter)
                else:
                    real_images, real_labels = next(data_iter)

            real_images, real_labels = real_images.to(device), real_labels.to(device)
            

            with torch.no_grad():
                fake_images = generator(z, fake_labels, evaluation=True)
                if conditional_strategy in ["ContraGAN", "Proxy_NCA_GAN", "XT_Xent_GAN"]:
                    _, _, dis_out_fake = discriminator(fake_images, fake_labels)
                    _, _, dis_out_real = discriminator(real_images, real_labels)
                elif conditional_strategy == "ACGAN":
                    _, dis_out_fake = discriminator(fake_images, fake_labels)
                    _, dis_out_real = discriminator(real_images, real_labels)
                elif conditional_strategy == "cGAN" or conditional_strategy == "no":
                    dis_out_fake = discriminator(fake_images, fake_labels)
                    dis_out_real = discriminator(real_images, real_labels)
                else:
                    raise NotImplementedError
                
                dis_out_fake = dis_out_fake.detach().cpu().numpy()
                dis_out_real = dis_out_real.detach().cpu().numpy()
                
                #print(dis_out_fake, dis_out_real)
                if weight_regularizer:
                    if reg_transform != None:
                        div_loss, softmax_output = weight_regularizer.loss(reg_transform(F.interpolate((fake_images + 1)/2, size=(224,224), mode='bilinear')), labels=True)
                    else:
                        div_loss, softmax_output = weight_regularizer.loss(fake_images , labels=True)
                    pred_class, pred_class_argmax = torch.max(softmax_output, 1)
                    
                    for argmax, val in zip(pred_class_argmax, pred_class):
                        if val > 0.9:
                            confidence_stats[argmax] = confidence_stats[argmax] + 1
                        
                    labels_clf = torch.argmax(softmax_output, dim = 1).cpu()
                    
                if conditional_strategy == "no":
                    fake_labels = labels_clf
                
                for i in range(fake_images.shape[0]):
                    generated_images.append(fake_images[i].detach())
                    generated_labels.append(fake_labels[i].detach().cpu().numpy())
                
            if batch_id == 0:
                confid = np.concatenate((dis_out_fake, dis_out_real), axis=0)
                confid_label = np.concatenate(([0.0]*len(dis_out_fake), [1.0]*len(dis_out_real)), axis=0)
            else:
                confid = np.concatenate((confid, dis_out_fake, dis_out_real), axis=0)
                confid_label = np.concatenate((confid_label, [0.0]*len(dis_out_fake), [1.0]*len(dis_out_real)), axis=0)


        

        total_images = [0] * num_classes
        if save:
            dataset_path = os.path.join(checkpoint_dir, "generated_dataset")
            print("Dataset is saved at:", dataset_path)
            if os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)
            
            os.makedirs(dataset_path)

            for (image, label) in tqdm(zip(generated_images, generated_labels)):

                #class_name = idx_to_class[label]
                class_path = os.path.join(dataset_path, str(label))
                if not os.path.exists(class_path):
                    os.makedirs(os.path.join(dataset_path, str(label)))
                
                total_images[label] += 1
                #print(image.shape)
                image_string = str(str(total_images[label])+".png")
                #print(image_string)
                save_image((image + 1)/2,os.path.join(class_path, image_string))

        real_confid = confid[confid_label==1.0]
        fake_confid = confid[confid_label==0.0]

        true_positive = real_confid[np.where(real_confid>cutoff)]
        true_negative = fake_confid[np.where(fake_confid<cutoff)]

        only_real_acc = len(true_positive)/len(real_confid)
        only_fake_acc = len(true_negative)/len(fake_confid)


        generator.train()
        discriminator.train()

        if weight_regularizer is not None:
            stats, kl_div = weight_regularizer.get_stats()
            stats = np.copy(stats)
            if logger is None:
                print("Class Statistics" + str(stats))
                print("Distribution of > 0.9 confidence samples" + str(confidence_stats/np.sum(confidence_stats)))
            else:
                logger.info("Class Statistics" + str(stats))
                logger.info("Distribution of > 0.9 confidence samples" + str(confidence_stats/np.sum(confidence_stats)))
            weight_regularizer.reset_stats()
            std_dev = np.std(stats)
        else:
            kl_div, std_dev = 0, 0

        return only_real_acc, only_fake_acc, kl_div, std_dev
    else:
        for batch_id in tqdm(range(total_batch)):
            try:
                if consistency_reg:
                    real_images, real_labels, real_images_aug = next(data_iter)
                else:
                    real_images, real_labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                if consistency_reg:
                    real_images, real_labels, real_images_aug = next(data_iter)
                else:
                    real_images, real_labels = next(data_iter)

            real_images, real_labels = real_images.to(device), real_labels.to(device)

            with torch.no_grad():
                if conditional_strategy in ["ContraGAN", "Proxy_NCA_GAN", "XT_Xent_GAN"]:
                    _, _, dis_out_real = discriminator(real_images, real_labels)
                elif conditional_strategy == "ACGAN":
                    _, dis_out_real = discriminator(real_images, real_labels)
                elif conditional_strategy == "cGAN" or conditional_strategy == "no":
                    dis_out_real = discriminator(real_images, real_labels)
                else:
                    raise NotImplementedError

                dis_out_real = dis_out_real.detach().cpu().numpy()

            if batch_id == 0:
                confid = dis_out_real
                confid_label = np.asarray([1.0]*len(dis_out_real), np.float32)
            else:
                confid = np.concatenate((confid, dis_out_real), axis=0)
                confid_label = np.concatenate((confid_label, [1.0]*len(dis_out_real)), axis=0)

        real_confid = confid[confid_label==1.0]
        true_positive = real_confid[np.where(real_confid>cutoff)]
        only_real_acc = len(true_positive)/len(real_confid)

        generator.train()
        discriminator.train()

        return only_real_acc

