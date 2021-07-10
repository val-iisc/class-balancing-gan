# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# trainer.py


from metrics.IS import calculate_incep_score
from metrics.FID import calculate_fid_score
#from utils.ada import augment
from utils.biggan_utils import toggle_grad, interp
from utils.sample import sample_latents, sample_1hot, make_mask, generate_images_for_KNN
from utils.plot import plot_img_canvas, plot_confidence_histogram, plot_img_canvas
from utils.utils import elapsed_time, calculate_all_sn, find_and_remove
from utils.losses import calc_derv4gp, calc_derv, latent_optimise
from utils.losses import Conditional_Contrastive_loss, Proxy_NCA_loss, XT_Xent_loss
from utils.diff_aug import DiffAugment
from utils.icr import ICR_Aug
from utils.calculate_accuracy import calculate_accuracy
from utils.utils import load_model_weights




import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import kornia.augmentation as K


import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

from os.path import join
import glob
from datetime import datetime
import shutil
import torch


SAVE_FORMAT = 'step={step:0>3}-Inception_mean={Inception_mean:<.4}-Inception_std={Inception_std:<.4}-FID={FID:<.5}.pth'

LOG_FORMAT = (
    "Step: {step:>7} "
    "Progress: {progress:<.1%} "
    "Elapsed: {elapsed} "
    "temperature: {temperature:<.6} "
    "ada_p: {ada_p:<.6} "
    "Discriminator_loss: {dis_loss:<.6} "
    "Generator_loss: {gen_loss:<.6} "
)


class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


class Trainer:
    def __init__(self, run_name, best_step, dataset_name, type4eval_dataset, logger, writer, n_gpus, gen_model, dis_model, inception_model, Gen_copy,
                 linear_model, Gen_ema, train_dataloader, eval_dataloader, conditional_strategy, pos_collected_numerator, z_dim, num_classes, hypersphere_dim,
                 d_spectral_norm, g_spectral_norm, G_optimizer, D_optimizer, L_optimizer, batch_size, g_steps_per_iter,  d_steps_per_iter, accumulation_steps,
                 total_step, G_loss, D_loss, contrastive_lambda, margin, tempering_type, tempering_step, start_temperature, end_temperature, gradient_penalty_for_dis,
                 gradient_penelty_lambda, weight_clipping_for_dis, weight_clipping_bound, consistency_reg, consistency_lambda, bcr, real_lambda, fake_lambda, zcr,
                 gen_lambda, dis_lambda, sigma_noise, diff_aug, ada, prev_ada_p, ada_target, ada_length, prior, truncated_factor, ema, latent_op,
                 latent_op_rate, latent_op_step, latent_op_step4eval, latent_op_alpha, latent_op_beta, latent_norm_reg_weight, default_device, print_every, save_every,
                 checkpoint_dir, evaluate, mu, sigma, best_fid, best_fid_checkpoint_path, mixed_precision, train_config, model_config, weight_regularizer, 
                 weight_regularizer_lambda, weight_regularizer_update, best_kl_div, ADA_cutoff, fixed_augment_p, lookahead):

        self.run_name = run_name
        self.best_step = best_step
        self.dataset_name = dataset_name
        self.type4eval_dataset = type4eval_dataset
        self.logger = logger
        self.writer = writer
        self.n_gpus = n_gpus

        self.gen_model = gen_model
        self.dis_model = dis_model
        self.inception_model = inception_model
        self.Gen_copy = Gen_copy
        self.linear_model = linear_model
        self.Gen_ema = Gen_ema

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.conditional_strategy = conditional_strategy
        self.pos_collected_numerator = pos_collected_numerator
        if self.pos_collected_numerator:
            assert self.conditional_strategy == "ContraGAN", "pos_collected_numerator option is not appliable except for ContraGAN."
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.hypersphere_dim = hypersphere_dim
        self.d_spectral_norm = d_spectral_norm
        self.g_spectral_norm = g_spectral_norm

        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.L_optimizer = L_optimizer
        self.batch_size = batch_size
        self.g_steps_per_iter = g_steps_per_iter
        self.d_steps_per_iter = d_steps_per_iter
        self.accumulation_steps = accumulation_steps
        self.total_step = total_step


        self.G_loss = G_loss
        self.D_loss = D_loss
        self.contrastive_lambda = contrastive_lambda
        self.margin = margin
        self.tempering_type = tempering_type
        self.tempering_step = tempering_step
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.gradient_penalty_for_dis = gradient_penalty_for_dis
        self.gradient_penelty_lambda = gradient_penelty_lambda
        self.weight_clipping_for_dis = weight_clipping_for_dis
        self.weight_clipping_bound = weight_clipping_bound
        self.consistency_reg = consistency_reg
        self.consistency_lambda = consistency_lambda
        self.bcr = bcr
        self.real_lambda = real_lambda
        self.fake_lambda = fake_lambda
        self.zcr = zcr
        self.gen_lambda = gen_lambda
        self.dis_lambda = dis_lambda
        self.sigma_noise = sigma_noise
        self.weighted_regularizer = weight_regularizer
        self.weighted_regularizer_lambda = weight_regularizer_lambda
        self.weighted_regularizer_update = weight_regularizer_update

        self.diff_aug = diff_aug
        self.ada = ada
        self.prev_ada_p = prev_ada_p
        self.fixed_augment_p = fixed_augment_p
        self.ada_target = ada_target
        self.ada_length = ada_length
        self.prior = prior
        self.truncated_factor = truncated_factor
        self.ema = ema
        self.latent_op = latent_op
        self.latent_op_rate = latent_op_rate
        self.latent_op_step = latent_op_step
        self.latent_op_step4eval = latent_op_step4eval
        self.latent_op_alpha = latent_op_alpha
        self.latent_op_beta = latent_op_beta
        self.latent_norm_reg_weight = latent_norm_reg_weight

        self.default_device = default_device
        self.print_every = print_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.evaluate = evaluate
        self.mu = mu
        self.sigma = sigma
        self.best_fid = best_fid
        self.best_kl_div = best_kl_div
        self.best_fid_checkpoint_path = best_fid_checkpoint_path
        self.mixed_precision = mixed_precision
        self.train_config = train_config
        self.model_config = model_config
        self.lookahead = lookahead

        self.start_time = datetime.now()
        self.l2_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        

        if self.conditional_strategy == 'ContraGAN':
            self.contrastive_criterion = Conditional_Contrastive_loss(self.default_device, self.batch_size, self.pos_collected_numerator)
            self.tempering_range = self.end_temperature - self.start_temperature
            assert tempering_type == "constant" or tempering_type == "continuous" or tempering_type == "discrete", \
                "tempering_type should be one of constant, continuous, or discrete"
            if self.tempering_type == 'discrete':
                self.tempering_interval = self.total_step//(self.tempering_step + 1)
        elif self.conditional_strategy == 'Proxy_NCA_GAN':
            if isinstance(self.dis_model, DataParallel):
                self.embedding_layer = self.dis_model.module.embedding
            else:
                self.embedding_layer = self.dis_model.embedding
            self.NCA_criterion = Proxy_NCA_loss(self.default_device, self.embedding_layer, self.num_classes, self.batch_size)
        elif self.conditional_strategy == 'XT_Xent_GAN':
            self.XT_Xent_criterion = XT_Xent_loss(self.default_device, self.batch_size)
        else:
            pass

        if self.conditional_strategy != "no":
            if self.dataset_name == "cifar10" or self.dataset_name == "lsun":
                sampler = "class_order_all"
            else:
                sampler = "class_order_some"
        else:
            sampler = "default"

        assert int(self.diff_aug)*int(self.ada) == 0, \
            "you can't simultaneously apply differentiable Augmentation (DiffAug) and adaptive augmentation (ADA)"

        self.policy = "color,translation,cutout"

        self.fixed_noise, self.fixed_fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1,
                                                                self.num_classes, None, self.default_device, sampler=sampler)

        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            

        if self.dataset_name == "imagenet" or self.dataset_name == "tiny_imagenet" :
            self.num_eval = {'train':50000, 'valid':50000, 'test':50000}
        elif self.dataset_name == "cifar10" or self.dataset_name == "lsun" or self.dataset_name == "inaturalist2019" or self.dataset_name == "cifar100":
            self.num_eval = {'train':50000, 'test':10000}
        
        if self.dataset_name == "inaturalist2019":
            
            mean = torch.Tensor([0.466, 0.471, 0.380])
            std = torch.Tensor([0.195, 0.194, 0.192])
            self.reg_transform = torch.nn.Sequential(K.Normalize(mean, std))
        else:
            self.reg_transform = None   
        

    #################################    proposed Contrastive Generative Adversarial Networks    ###################################
    ################################################################################################################################
    def run_ours(self, current_step, total_step):
        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()

        self.logger.info('Start training...')
        step_count = current_step
        train_iter = iter(self.train_dataloader)

        if self.ada:
            self.ada_augment = torch.tensor([0.0, 0.0], device = self.default_device)
            if self.prev_ada_p is not None:
                self.ada_aug_p = self.prev_ada_p
            else:
                self.ada_aug_p = self.fixed_augment_p if self.fixed_augment_p > 0.0 else 0.0
            self.ada_aug_step = self.ada_target/self.ada_length
        else:
            self.ada_aug_p = 'No'
        while step_count <= total_step:
            # ================== TRAIN D ================== #
            if self.tempering_type == 'continuous':
                t = self.start_temperature + step_count*(self.end_temperature - self.start_temperature)/total_step
            elif self.tempering_type == 'discrete':
                t = self.start_temperature + \
                    (step_count//self.tempering_interval)*(self.end_temperature-self.start_temperature)/self.tempering_step
            else:
                t = self.start_temperature

            toggle_grad(self.dis_model, True)
            toggle_grad(self.gen_model, False)
            for step_index in range(self.d_steps_per_iter):
                self.D_optimizer.zero_grad()
                for acml_step in range(self.accumulation_steps):
                    with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mp:
                        try:
                            if self.consistency_reg or self.conditional_strategy == "XT_Xent_GAN":
                                real_images, real_labels, real_images_aug = next(train_iter)
                            else:
                                real_images, real_labels = next(train_iter)
                        except StopIteration:
                            train_iter = iter(self.train_dataloader)
                            if self.consistency_reg or self.conditional_strategy == "XT_Xent_GAN":
                                real_images, real_labels, real_images_aug = next(train_iter)
                            else:
                                real_images, real_labels = next(train_iter)

                    
                        real_images, real_labels = real_images.to(self.default_device), real_labels.to(self.default_device)
                        if self.diff_aug:
                            real_images = DiffAugment(real_images, policy=self.policy)

                        if self.ada:
                            real_images, _ = augment(real_images, self.ada_aug_p)

                        if self.zcr:
                            z, fake_labels, z_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                 self.sigma_noise, self.default_device)
                        else:
                            z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                            None, self.default_device)
                            

                        real_cls_mask = make_mask(real_labels, self.num_classes, self.default_device)

                        cls_proxies_real, cls_embed_real, dis_out_real = self.dis_model(real_images, real_labels)

                        fake_images = self.gen_model(z, fake_labels)
                        
                        if self.diff_aug:
                            fake_images = DiffAugment(fake_images, policy=self.policy)

                        if self.ada:
                            fake_images, _ = augment(fake_images, self.ada_aug_p)

                        cls_proxies_fake, cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                        dis_acml_loss = self.D_loss(dis_out_real, dis_out_fake)

                        if self.bcr:
                            real_images_aug = ICR_Aug(real_images)
                            fake_images_aug = ICR_Aug(fake_images)
                            cls_proxies_real_aug, cls_embed_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                            cls_proxies_fake_aug, cls_embed_fake_aug, dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                            bcr_real_loss = self.l2_loss(dis_out_real, dis_out_real_aug)
                            bcr_fake_loss = self.l2_loss(dis_out_fake, dis_out_fake_aug)
                            dis_acml_loss += self.real_lambda*bcr_real_loss + self.fake_lambda*bcr_fake_loss

                        if self.zcr:
                            fake_images_zaug = self.gen_model(z_t, fake_labels)
                            cls_proxies_fake_zaug, cls_embed_fake_zaug, dis_out_fake_zaug = self.dis_model(fake_images_zaug, fake_labels)
                            zcr_dis_loss = self.l2_loss(dis_out_fake, dis_out_fake_zaug)
                            dis_acml_loss += self.dis_lambda*zcr_dis_loss

                        if self.conditional_strategy == "ContraGAN":
                            dis_acml_loss += self.contrastive_lambda*self.contrastive_criterion(cls_embed_real, cls_proxies_real, real_cls_mask,
                                                                                            real_labels, t, self.margin)
                        elif self.conditional_strategy == "Proxy_NCA_GAN":
                            dis_acml_loss += self.contrastive_lambda*self.NCA_criterion(cls_embed_real, cls_proxies_real, real_labels)
                        elif self.conditional_strategy == "XT_Xent_GAN":
                            pass
                        else:
                            raise NotImplementedError

                        if self.ada and self.fixed_augment_p == 0.0:
                            ada_aug_data = torch.tensor((torch.sign(dis_out_real).sum().item(), dis_out_real.shape[0]),
                                                        device = self.default_device)
                            self.ada_augment += ada_aug_data
                            if self.ada_augment[1] > (self.batch_size*4 - 1):
                                authen_out_signs, num_outputs = self.ada_augment.tolist()
                                r_t_stat = authen_out_signs/num_outputs
                                sign = 1 if r_t_stat > self.ada_target else -1
                                self.ada_aug_p += sign*self.ada_aug_step*num_outputs
                                self.ada_aug_p = min(1.0, max(0.0, self.ada_aug_p))
                                self.ada_augment.mul_(0.0)

                        if self.consistency_reg or self.conditional_strategy == "XT_Xent_GAN":
                            real_images_aug = real_images_aug.to(self.default_device)
                            _, cls_real_aug_embed, dis_real_aug_authen_out = self.dis_model(real_images_aug, real_labels)
                            if self.conditional_strategy == "XT_Xent_GAN":
                                dis_acml_loss += self.contrastive_lambda*self.XT_Xent_criterion(cls_embed_real, cls_real_aug_embed, t)
                            if self.consistency_reg:
                                consistency_loss = self.l2_loss(dis_out_real, dis_real_aug_authen_out)
                                dis_acml_loss += self.consistency_lambda*consistency_loss

                        dis_acml_loss = dis_acml_loss/self.accumulation_steps

                    if self.mixed_precision:
                        self.scaler.scale(dis_acml_loss).backward()
                    else:
                        dis_acml_loss.backward()
                
                if self.mixed_precision:
                    self.scaler.step(self.D_optimizer)
                    self.scaler.update()
                else:
                    self.D_optimizer.step()

                if self.weight_clipping_for_dis:
                    for p in self.dis_model.parameters():
                        p.data.clamp_(-self.weight_clipping_bound, self.weight_clipping_bound)

            if step_count % self.print_every == 0 and step_count !=0 and self.logger:
                if self.d_spectral_norm:
                    dis_sigmas = calculate_all_sn(self.dis_model)
                    self.writer.add_scalars('SN_of_dis', dis_sigmas, step_count)

            # ================== TRAIN G ================== #
            toggle_grad(self.dis_model, False)
            toggle_grad(self.gen_model, True)
            for step_index in range(self.g_steps_per_iter):
                self.G_optimizer.zero_grad()
                for acml_step in range(self.accumulation_steps):
                    with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mp:
                        if self.zcr:
                            z, fake_labels, z_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                 self.sigma_noise, self.default_device)
                        else:
                            z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                            None, self.default_device)

                        fake_cls_mask = make_mask(fake_labels, self.num_classes, self.default_device)

                        fake_images = self.gen_model(z, fake_labels)
                        if self.diff_aug:
                            fake_images = DiffAugment(fake_images, policy=self.policy)

                        if self.ada:
                            fake_images, _ = augment(fake_images, self.ada_aug_p)
                        cls_proxies_fake, cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                        

                        cls_proxies_real, cls_embed_real, dis_out_real = self.dis_model(real_images, real_labels)
                        gen_acml_loss = self.G_loss(dis_out_fake, dis_out_real)

                        if self.zcr:
                            fake_images_zaug = self.gen_model(z_t, fake_labels)
                            zcr_gen_loss = -1 * self.l2_loss(fake_images, fake_images_zaug)
                            gen_acml_loss += self.gen_lambda*zcr_gen_loss

                        if self.conditional_strategy == "ContraGAN":
                            gen_acml_loss += self.contrastive_lambda*self.contrastive_criterion(cls_out_fake, cls_proxies_fake, fake_cls_mask, fake_labels, t, 0.0)
                        elif self.conditional_strategy == "Proxy_NCA_GAN":
                            gen_acml_loss += self.contrastive_lambda*self.NCA_criterion(cls_out_fake, cls_proxies_fake, fake_labels)
                        elif self.conditional_strategy == "XT_Xent_GAN":
                            pass
                        else:
                            raise NotImplementedError

                        gen_acml_loss = gen_acml_loss/self.accumulation_steps

                    if self.mixed_precision:
                        self.scaler.scale(gen_acml_loss).backward()
                    else:
                        gen_acml_loss.backward()

                if self.mixed_precision:
                    self.scaler.step(self.G_optimizer)
                    self.scaler.update()
                else:
                    self.G_optimizer.step()

                # if ema is True: we update parameters of the Gen_copy in adaptive way.
                if self.ema:
                    self.Gen_ema.update(step_count)

                step_count += 1
            if step_count % self.print_every == 0 and self.logger:
                log_message = LOG_FORMAT.format(step=step_count,
                                                progress=step_count/total_step,
                                                elapsed=elapsed_time(self.start_time),
                                                temperature=t,
                                                ada_p=self.ada_aug_p,
                                                dis_loss=dis_acml_loss.item(),
                                                gen_loss=gen_acml_loss.item(),
                                                )
                self.logger.info(log_message)

                if self.g_spectral_norm:
                    gen_sigmas = calculate_all_sn(self.gen_model)
                    self.writer.add_scalars('SN_of_gen', gen_sigmas, step_count)

                self.writer.add_scalars('Losses', {'discriminator': dis_acml_loss.item(),
                                                   'generator': gen_acml_loss.item()}, step_count)
                if self.ada:
                    self.writer.add_scalar('ada_p', self.ada_aug_p, step_count)

                with torch.no_grad():
                    self.gen_model.eval()
                    if self.Gen_copy is not None:
                        self.Gen_copy.train()
                    generator = self.Gen_copy if self.Gen_copy is not None else self.gen_model

                    generated_images = generator(self.fixed_noise, self.fixed_fake_labels)
                    self.writer.add_images('Generated samples', (generated_images+1)/2, step_count)
                    self.gen_model.train()

            if step_count % self.save_every == 0 or step_count == total_step:
                if self.evaluate:
                    is_best = self.evaluation(step_count)
                    self.save(step_count, is_best)
                else:
                    self.save(step_count, False)
        return step_count-1
    ################################################################################################################################


    #######################    dcgan/resgan/wgan_wc/wgan_gp/sngan/sagan/biggan/logan/crgan implementations    ######################
    ################################################################################################################################
    def run(self, current_step, total_step):
        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()

        self.logger.info('Start training...')
        step_count = current_step
        train_iter = iter(self.train_dataloader)

        if self.ada:
            self.ada_augment = torch.tensor([0.0, 0.0], device = self.default_device)
            if self.prev_ada_p is not None:
                self.ada_aug_p = self.prev_ada_p
            else:
                self.ada_aug_p = self.fixed_augment_p if self.fixed_augment_p > 0.0 else 0.0
            self.ada_aug_step = self.ada_target/self.ada_length
        else:
            self.ada_aug_p = 'No'
        while step_count <= total_step:
            # ================== TRAIN D ================== #
            toggle_grad(self.dis_model, True)
            toggle_grad(self.gen_model, False)
            for step_index in range(self.d_steps_per_iter):
                self.D_optimizer.zero_grad()
                for acml_index in range(self.accumulation_steps):
                    
                    try:
                        if self.consistency_reg:
                            real_images, real_labels, real_images_aug = next(train_iter)
                        else:
                            real_images, real_labels = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_dataloader)
                        if self.consistency_reg:
                            real_images, real_labels, real_images_aug = next(train_iter)
                        else:
                            real_images, real_labels = next(train_iter)

                    with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mp:
                        real_images, real_labels = real_images.to(self.default_device), real_labels.to(self.default_device)
                        if self.diff_aug:
                            real_images = DiffAugment(real_images, policy=self.policy)

                        if self.ada:
                            real_images, _ = augment(real_images, self.ada_aug_p)

                        if self.zcr:
                            z, fake_labels, z_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                 self.sigma_noise, self.default_device)
                        else:
                            z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                            None, self.default_device)
                            
                            
                        
                        # if self.conditional_strategy == "cGAN":
                        #      fake_labels = real_labels

                        if self.latent_op:
                            z = latent_optimise(z, fake_labels, self.gen_model, self.dis_model, self.latent_op_step, self.latent_op_rate,
                                                self.latent_op_alpha, self.latent_op_beta, False, self.default_device)

                        fake_images = self.gen_model(z, fake_labels)

                        #print(fake_images.shape)
                        if self.diff_aug:
                            fake_images = DiffAugment(fake_images, policy=self.policy)

                        if self.ada:
                            fake_images, _ = augment(fake_images, self.ada_aug_p)

                        if self.conditional_strategy == "ACGAN":
                            cls_out_real, dis_out_real = self.dis_model(real_images, real_labels)
                            cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                        elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                            dis_out_real = self.dis_model(real_images, real_labels)
                            dis_out_fake = self.dis_model(fake_images, fake_labels)
                        else:
                            raise NotImplementedError

                        dis_acml_loss = self.D_loss(dis_out_real, dis_out_fake)

                        if self.bcr:
                            real_images_aug = ICR_Aug(real_images)
                            fake_images_aug = ICR_Aug(fake_images)
                            if self.conditional_strategy == "ACGAN":
                                cls_out_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                cls_out_fake_aug, dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                            elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                                dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                            else:
                                raise NotImplementedError

                            bcr_real_loss = self.l2_loss(dis_out_real, dis_out_real_aug)
                            bcr_fake_loss = self.l2_loss(dis_out_fake, dis_out_fake_aug)
                            dis_acml_loss += self.real_lambda*bcr_real_loss + self.fake_lambda*bcr_fake_loss

                        if self.zcr:
                            fake_images_zaug = self.gen_model(z_t, fake_labels)
                            if self.conditional_strategy == "ACGAN":
                                cls_out_fake_zaug, dis_out_fake_zaug = self.dis_model(fake_images_zaug, fake_labels)
                            elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                                dis_out_fake_zaug = self.dis_model(fake_images_zaug, fake_labels)
                            else:
                                raise NotImplementedError

                            zcr_dis_loss = self.l2_loss(dis_out_fake, dis_out_fake_zaug)
                            dis_acml_loss += self.dis_lambda*zcr_dis_loss

                        if self.conditional_strategy == "ACGAN":
                            dis_acml_loss += (self.ce_loss(cls_out_real, real_labels) + self.ce_loss(cls_out_fake, fake_labels))

                        if self.gradient_penalty_for_dis:
                            dis_acml_loss += gradient_penelty_lambda*calc_derv4gp(self.dis_model, real_images, fake_images, real_labels, self.default_device)

                        if self.ada and self.fixed_augment_p == 0.0:
                            ada_aug_data = torch.tensor((torch.sign(dis_out_real).sum().item(), dis_out_real.shape[0]),
                                                        device = self.default_device)
                            self.ada_augment += ada_aug_data
                            if self.ada_augment[1] > (self.batch_size*4 - 1):
                                authen_out_signs, num_outputs = self.ada_augment.tolist()
                                r_t_stat = authen_out_signs/num_outputs
                                sign = 1 if r_t_stat > self.ada_target else -1
                                self.ada_aug_p += sign*self.ada_aug_step*num_outputs
                                self.ada_aug_p = min(1.0, max(0.0, self.ada_aug_p))
                                self.ada_augment.mul_(0.0)

                        if self.consistency_reg:
                            real_images_aug = real_images_aug.to(self.default_device)
                            if self.conditional_strategy == "ACGAN":
                                cls_out_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                            elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                                dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                            else:
                                raise NotImplementedError
                            consistency_loss = self.l2_loss(dis_out_real, dis_out_real_aug)
                            dis_acml_loss += self.consistency_lambda*consistency_loss
                    
                        

                        dis_acml_loss = dis_acml_loss/self.accumulation_steps

                    if self.mixed_precision:
                        self.scaler.scale(dis_acml_loss).backward()
                    else:
                        dis_acml_loss.backward()

                
                if self.mixed_precision:
                    self.scaler.step(self.D_optimizer)
                    self.scaler.update()
                else:
                    self.D_optimizer.step()

                if self.weight_clipping_for_dis:
                    for p in self.dis_model.parameters():
                        p.data.clamp_(-self.weight_clipping_bound, self.weight_clipping_bound)

            if step_count % self.print_every == 0 and step_count !=0 and self.logger:
                if self.d_spectral_norm:
                    dis_sigmas = calculate_all_sn(self.dis_model)
                    self.writer.add_scalars('SN_of_dis', dis_sigmas, step_count)

            # ================== TRAIN G ================== #
            toggle_grad(self.dis_model, False)
            toggle_grad(self.gen_model, True)
            for step_index in range(self.g_steps_per_iter):
                self.G_optimizer.zero_grad()
                for acml_step in range(self.accumulation_steps):
                    with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mp:
                        if self.zcr:
                            z, fake_labels, z_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                 self.sigma_noise, self.default_device)
                        else:
                            z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                            None, self.default_device)
                            #inv_weight = [1/i for i in self.weighted_regularizer.count_class_samples]
                            #fake_labels = torch.tensor(list(torch.utils.data.WeightedRandomSampler(inv_weight, self.batch_size))).to(self.default_device)
                        

                        if self.latent_op:
                            z, transport_cost = latent_optimise(z, fake_labels, self.gen_model, self.dis_model, self.latent_op_step, self.latent_op_rate,
                                                                self.latent_op_alpha, self.latent_op_beta, True, self.default_device)
                        

                        # if self.conditional_strategy == "cGAN":
                        #     fake_labels = real_labels

                        
                        fake_images = self.gen_model(z, fake_labels)

                        gen_acml_loss = 0
                        if self.weighted_regularizer is not None and self.weighted_regularizer_lambda > 0:
                            if self.dataset_name == "inaturalist2019":
                                gen_acml_loss += self.weighted_regularizer_lambda * self.weighted_regularizer.loss(self.reg_transform(F.interpolate((fake_images + 1)/2, size=(224,224), mode='bilinear')), writer = self.writer, target_labels=fake_labels)
                            elif self.weighted_regularizer.pretrained_model.__class__.__name__ == "ResNetV2":
                                gen_acml_loss += self.weighted_regularizer_lambda * self.weighted_regularizer.loss(fake_images)
                            else:
                                gen_acml_loss += self.weighted_regularizer_lambda * self.weighted_regularizer.loss(fake_images)           

                            #gen_acml_loss += 0.5 * self.ce_loss(self.weighted_regularizer.pretrained_model(self.reg_transform(F.interpolate((fake_images + 1)/2, size=(224,224), mode='bilinear'))), fake_labels)            

                        if self.diff_aug:
                            fake_images = DiffAugment(fake_images, policy=self.policy)

                        if self.ada:
                            fake_images, _ = augment(fake_images, self.ada_aug_p)

                        if self.conditional_strategy == "ACGAN":
                            cls_out_real, dis_out_real = self.dis_model(real_images, real_labels)
                            cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                        elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                            dis_out_real = self.dis_model(real_images, real_labels)
                            dis_out_fake = self.dis_model(fake_images, fake_labels)
                        else:
                            raise NotImplementedError
                        


                        gen_acml_loss += self.G_loss(dis_out_fake, dis_out_real)

                        if self.zcr:
                            fake_images_zaug = self.gen_model(z_t, fake_labels)
                            zcr_gen_loss = -1 * self.l2_loss(fake_images, fake_images_zaug)
                            gen_acml_loss += self.gen_lambda*zcr_gen_loss

                        if self.conditional_strategy == "ACGAN":
                            gen_acml_loss += self.ce_loss(cls_out_fake, fake_labels)

                        if self.latent_op:
                            gen_acml_loss += transport_cost*self.latent_norm_reg_weight
                        gen_acml_loss = gen_acml_loss/self.accumulation_steps



                            # Add cross entropy loss for testing the classifier hypothesis
                            #gen_acml_loss += self.ce_loss(self.weighted_regularizer.pretrained_model((fake_images)), fake_labels)

                        

                    if self.mixed_precision:
                        self.scaler.scale(gen_acml_loss).backward()
                    else:
                        gen_acml_loss.backward()

                if self.mixed_precision:
                    self.scaler.step(self.G_optimizer)
                    self.scaler.update()
                else:
                    self.G_optimizer.step()

                if self.lookahead and (step_count) % self.G_optimizer.k == 0:
                    self.G_optimizer.update_lookahead()
                    self.D_optimizer.update_lookahead()

                # if ema is True: we update parameters of the Gen_copy in adaptive way.
                if self.ema:
                    self.Gen_ema.update(step_count)

                step_count += 1

            if step_count % self.print_every == 0 and self.logger:
                log_message = LOG_FORMAT.format(step=step_count,
                                                progress=step_count/total_step,
                                                elapsed=elapsed_time(self.start_time),
                                                temperature='No',
                                                ada_p=self.ada_aug_p,
                                                dis_loss=dis_acml_loss.item(),
                                                gen_loss=gen_acml_loss.item(),
                                                )
                self.logger.info(log_message)

                if self.g_spectral_norm:
                    gen_sigmas = calculate_all_sn(self.gen_model)
                    self.writer.add_scalars('SN_of_gen', gen_sigmas, step_count)

                self.writer.add_scalars('Losses', {'discriminator': dis_acml_loss.item(),
                                                   'generator': gen_acml_loss.item()}, step_count)
                if self.ada:
                    self.writer.add_scalars('ada_p', self.ada_aug_p, step_count)

                with torch.no_grad():
                    self.gen_model.eval()
                    if self.Gen_copy is not None:
                        self.Gen_copy.train()
                    generator = self.Gen_copy if self.Gen_copy is not None else self.gen_model
                    
                    generated_images = generator(self.fixed_noise, self.fixed_fake_labels)
                    self.writer.add_images('Generated samples', (generated_images+1)/2, step_count)
                    self.gen_model.train()

            if step_count % self.weighted_regularizer_update == 0 or step_count == total_step:
                if self.weighted_regularizer is not None:
                    if  step_count % (20 * self.weighted_regularizer_update) == 0:
                        self.weighted_regularizer.update(self.logger)
                    else:
                        self.weighted_regularizer.update()
            

            if step_count % self.save_every == 0 or step_count == total_step:
                if self.evaluate:
                    is_best = self.evaluation(step_count)
                    self.save(step_count, is_best)
                else:
                    self.save(step_count, False)

            
        return step_count-1
    ################################################################################################################################


    ################################################################################################################################
    def save(self, step, is_best):
        when = "best" if is_best is True else "current"
        self.dis_model.eval()
        self.gen_model.eval()
        if self.Gen_copy is not None:
            self.Gen_copy.eval()

        g_states = {'seed': self.train_config['seed'], 'run_name': self.run_name, 'step': step, 'best_step': self.best_step,
                    'state_dict': self.gen_model.state_dict(), 'optimizer': self.G_optimizer.state_dict(), 'ada_p': self.ada_aug_p}

        d_states = {'seed': self.train_config['seed'], 'run_name': self.run_name, 'step': step, 'best_step': self.best_step,
                    'state_dict': self.dis_model.state_dict(), 'optimizer': self.D_optimizer.state_dict(), 'ada_p': self.ada_aug_p,
                    'best_fid': self.best_fid, 'best_fid_checkpoint_path': self.checkpoint_dir}

        if len(glob.glob(join(self.checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))) >= 1:
            find_and_remove(glob.glob(join(self.checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))[0])
            find_and_remove(glob.glob(join(self.checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0])

        g_checkpoint_output_path = join(self.checkpoint_dir, "model=G-{when}-weights-step={step}.pth".format(when=when, step=str(step)))
        d_checkpoint_output_path = join(self.checkpoint_dir, "model=D-{when}-weights-step={step}.pth".format(when=when, step=str(step)))

        if when == "best":
            if len(glob.glob(join(self.checkpoint_dir,"model=G-current-weights-step*.pth".format(when=when)))) >= 1:
                find_and_remove(glob.glob(join(self.checkpoint_dir,"model=G-current-weights-step*.pth".format(when=when)))[0])
                find_and_remove(glob.glob(join(self.checkpoint_dir,"model=D-current-weights-step*.pth".format(when=when)))[0])

            g_checkpoint_output_path_ = join(self.checkpoint_dir, "model=G-current-weights-step={step}.pth".format(when=when, step=str(step)))
            d_checkpoint_output_path_ = join(self.checkpoint_dir, "model=D-current-weights-step={step}.pth".format(when=when, step=str(step)))

            torch.save(g_states, g_checkpoint_output_path_)
            torch.save(d_states, d_checkpoint_output_path_)

        torch.save(g_states, g_checkpoint_output_path)
        torch.save(d_states, d_checkpoint_output_path)

        if self.Gen_copy is not None:
            g_ema_states = {'state_dict': self.Gen_copy.state_dict()}
            if len(glob.glob(join(self.checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))) >= 1:
                find_and_remove(glob.glob(join(self.checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0])

            g_ema_checkpoint_output_path = join(self.checkpoint_dir, "model=G_ema-{when}-weights-step={step}.pth".format(when=when, step=str(step)))

            if when == "best":
                if len(glob.glob(join(self.checkpoint_dir,"model=G_ema-current-weights-step*.pth".format(when=when)))) >= 1:
                    find_and_remove(glob.glob(join(self.checkpoint_dir,"model=G_ema-current-weights-step*.pth".format(when=when)))[0])

                g_ema_checkpoint_output_path_ = join(self.checkpoint_dir, "model=G_ema-current-weights-step={step}.pth".format(when=when, step=str(step)))

                torch.save(g_ema_states, g_ema_checkpoint_output_path_)

            torch.save(g_ema_states, g_ema_checkpoint_output_path)

        if self.logger:
            self.logger.info("Saved model to {}".format(self.checkpoint_dir))

        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()
    ################################################################################################################################


    ################################################################################################################################
    def evaluation(self, step, save = False, evaluation_checkpoint=None):

        with torch.no_grad():
            self.logger.info("Start Evaluation ({step} Step): {run_name}".format(step=step, run_name=self.run_name))
            is_best = False
            num_random_restarts = 3
            self.dis_model.eval()
            self.gen_model.eval()
            if self.Gen_copy is not None:
                self.Gen_copy.train()
            generator = self.Gen_copy if self.Gen_copy is not None else self.gen_model

            if self.latent_op:
                self.fixed_noise = latent_optimise(self.fixed_noise, self.fixed_fake_labels, generator, self.dis_model, self.latent_op_step, self.latent_op_rate,
                                                   self.latent_op_alpha, self.latent_op_beta, False, self.default_device)
            
            

            fid_score, self.m1, self.s1 = calculate_fid_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, self.num_eval[self.type4eval_dataset],
                                                              self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                              self.latent_op_beta, self.default_device, self.mu, self.sigma, self.run_name)

            kl_score, kl_std = calculate_incep_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, self.num_eval[self.type4eval_dataset],
                                        self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                        self.latent_op_beta, 10, self.default_device)
            


            

            ### pre-calculate an inception score
            ### calculating inception score using the below will give you an underestimated one.
            ### plz use the official tensorflow implementation(inception_tensorflow.py).
            real_train_acc, fake_acc, kl_div_clf, std_dev = calculate_accuracy(self.train_dataloader, generator, self.dis_model, self.D_loss, 100000, self.truncated_factor, self.prior,
                                                self.latent_op,self.latent_op_step, self.latent_op_alpha,self. latent_op_beta, self.default_device,
                                                consistency_reg=self.consistency_reg, eval_generated_sample=True, weight_regularizer=self.weighted_regularizer, save=save, checkpoint_dir=self.checkpoint_dir, logger = self.logger, reg_transform=self.reg_transform)
            # Disabling this temporarilly
            if evaluation_checkpoint is not None:
                # Do Random restarts and generate FID scores
                fid_scores, kl_div_scores = np.zeros(num_random_restarts), np.zeros(num_random_restarts)
                #load_model_weights(self.weighted_regularizer.pretrained_model, evaluation_checkpoint)

                for i in range(num_random_restarts):
                    fid_scores[i], _, _ = calculate_fid_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, 50000,
                                                    self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                    self.latent_op_beta, self.default_device, self.mu, self.sigma, self.run_name)
                    _, _, kl_div_scores[i], _ = calculate_accuracy(self.train_dataloader, generator, self.dis_model, self.D_loss, 50000, self.truncated_factor, self.prior,
                                    self.latent_op,self.latent_op_step, self.latent_op_alpha,self. latent_op_beta, self.default_device,
                                    consistency_reg=self.consistency_reg, eval_generated_sample=True, weight_regularizer=self.weighted_regularizer, save=False, checkpoint_dir=self.checkpoint_dir, logger=self.logger, reg_transform=self.reg_transform)
                    
                self.logger.info("KL DIV score using Using {type} moments score: Avg Scores: {scr} Std Dev: {std_dev}".format(type = self.type4eval_dataset, scr = str(np.mean(kl_div_scores)), std_dev = str(np.std(kl_div_scores))))
                self.logger.info("FID  score using Using {type} moments score: Avg Scores: {scr} Std Dev: {std_dev}".format(type = self.type4eval_dataset, scr = str(np.mean(fid_scores)), std_dev = str(np.std(fid_scores))))

            
            if evaluation_checkpoint is not None and self.dataset_name != "inaturalist2019":
                
                fid_score_minority, minority_mu, minority_sig = calculate_fid_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, 50000,
                                                              self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                              self.latent_op_beta, self.default_device, None, None, self.run_name, sampler = "minority", pretrained_classifier=self.weighted_regularizer.pretrained_model)
                fid_score_majority, majority_mu, mmajority_sig = calculate_fid_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, 50000,
                                                self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                self.latent_op_beta, self.default_device, None, None, self.run_name, sampler = "majority", pretrained_classifier=self.weighted_regularizer.pretrained_model)





            if self.type4eval_dataset == 'train':
                acc_dict = {'real_train': real_train_acc, 'fake': fake_acc, 'kl_div_clf': kl_div_clf, 'std_dev': std_dev}
            else:
                real_eval_acc = calculate_accuracy(self.eval_dataloader, generator, self.dis_model, self.D_loss, 10000, self.truncated_factor, self.prior,
                                                   self.latent_op, self.latent_op_step, self.latent_op_alpha,self. latent_op_beta, self.default_device,
                                                   consistency_reg=False, eval_generated_sample=False)
                acc_dict = {'real_train': real_train_acc, 'real_valid': real_eval_acc, 'fake': fake_acc, 'kl_div_clf': kl_div_clf, 'std_dev': std_dev}

            self.writer.add_scalars('Accuracy', acc_dict, step)
            

            if self.best_fid is None:
                self.best_fid, self.best_step, is_best, self.best_kl_div = fid_score, step, True, kl_div_clf
                
            else:
                if fid_score <= self.best_fid:
                    self.best_fid, self.best_step, is_best, self.best_kl_div = fid_score, step, True, kl_div_clf

            self.writer.add_scalars('FID score', {'using {type} moments'.format(type=self.type4eval_dataset):fid_score}, step)
            self.writer.add_scalars('IS score', {'{num} generated images'.format(num=str(self.num_eval[self.type4eval_dataset])):kl_score}, step)
            self.logger.info('FID score (Step: {step}, Using {type} moments): {FID}'.format(step=step, type=self.type4eval_dataset, FID=fid_score))
            self.logger.info('KLDIV score (Step: {step}, Using {type} moments): {KLDIV}'.format(step=step, type=self.type4eval_dataset, KLDIV=kl_div_clf))
            self.logger.info('Inception score (Step: {step}, {num} generated images): {IS}'.format(step=step, num=str(self.num_eval[self.type4eval_dataset]), IS=kl_score))
            self.logger.info('Best FID score (Step: {step}, Using {type} moments): {FID}'.format(step=self.best_step, type=self.type4eval_dataset, FID=self.best_fid))
            self.logger.info('KL DIV Score at best FID step score (Step: {step}, Using {type} moments): {KLDIV}'.format(step=self.best_step, type=self.type4eval_dataset, KLDIV=self.best_kl_div))

            
            self.dis_model.train()
            self.gen_model.train()

        return is_best
    ################################################################################################################################

    ################################################################################################################################
    def Nearest_Neighbor(self, nrow, ncol):
        with torch.no_grad():
            self.gen_model.eval()
            if self.Gen_copy is not None:
                self.Gen_copy.train()
            generator = self.Gen_copy if self.Gen_copy is not None else self.gen_model

            resnet50_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
            resnet50_conv = nn.Sequential(*list(resnet50_model.children())[:-1]).to(self.default_device)
            if self.n_gpus > 1:
                resnet50_conv = DataParallel(resnet50_conv, output_device=self.default_device)
            resnet50_conv.eval()
            with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mp:
                self.logger.info("Start Nearest Neighbor....")
                for c in tqdm(range(self.num_classes)):
                    fake_images, fake_labels = generate_images_for_KNN(self.batch_size, c, generator, self.dis_model, self.truncated_factor, self.prior, self.latent_op,
                                                                    self.latent_op_step, self.latent_op_alpha, self.latent_op_beta, self.default_device)

                    if self.conditional_strategy == "no":
                        if self.dataset_name == "inaturalist2019":
                            loss, labels = self.weighted_regularizer.loss(self.reg_transform((fake_images + 1)/2), True)
                        else:
                            loss, labels = self.weighted_regularizer.loss(fake_images, True)
                        inc_classes = [c]
                        fake_images = torch.from_numpy(fake_images.cpu().numpy()[np.isin(labels,inc_classes)]).to(self.default_device)

                    
                    fake_image = torch.unsqueeze(fake_images[1], dim=0)

                    fake_anchor_embedding = torch.squeeze(resnet50_conv((fake_image+1)/2))

                    hit = 0
                    eval_iter = iter(self.eval_dataloader)
                    for batch_idx in range(len(eval_iter)):
                        if self.consistency_reg:
                            real_images, real_labels, real_images_aug = next(eval_iter)
                        else:
                            real_images, real_labels = next(eval_iter)

                        num_cls_in_batch = 10
                        if num_cls_in_batch > 0:
                            #real_images = real_images[torch.tensor(real_labels.detach().cpu().numpy() == c)]
                            real_images = real_images.to(self.default_device)

                            if hit == 0:
                                real_embeddings = torch.squeeze(resnet50_conv((real_images+1)/2))
                            
                                if num_cls_in_batch == 1:
                                    distances = torch.square(real_embeddings - fake_anchor_embedding).mean(dim=0).detach().cpu().numpy()
                                else:
                                    distances = torch.square(real_embeddings - fake_anchor_embedding).mean(dim=1).detach().cpu().numpy()
                                
                                holder = real_images.detach().cpu().numpy()
                                hit += 1
                            else:
                                real_embeddings = torch.squeeze(resnet50_conv((real_images+1)/2))
                                if num_cls_in_batch == 1:
                                    distances = np.append(distances, torch.square(real_embeddings - fake_anchor_embedding).mean(dim=0).detach().cpu().numpy())
                                else:
                                    distances = np.concatenate([distances, torch.square(real_embeddings - fake_anchor_embedding).mean(dim=1).detach().cpu().numpy()], axis=0)
                                holder = np.concatenate([holder, real_images.detach().cpu().numpy()], axis=0)
                            
                        else:
                            pass
                    
                    
                    nearest_indices = (-distances).argsort()[-(ncol-1):][::-1]
                    if c % nrow == 0:
                        canvas = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                    elif c % nrow == nrow-1:
                        row_images = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                        canvas = np.concatenate((canvas, row_images), axis=0)
                        plot_img_canvas((torch.from_numpy(canvas)+1)/2, "./figures/{run_name}/Fake_anchor_{ncol}NN_{cls}.png".\
                                        format(run_name=self.run_name,ncol=ncol, cls=c), self.logger, ncol)
                    else:
                        row_images = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                        canvas = np.concatenate((canvas, row_images), axis=0)

        self.gen_model.train()
    ################################################################################################################################

    def linear_interpolation(self, nrow, ncol, fix_z, fix_y):
        self.gen_model.eval()
        if self.Gen_copy is not None:
            self.Gen_copy.train()
        generator = self.Gen_copy if self.Gen_copy is not None else self.gen_model
        assert int(fix_z)*int(fix_y) != 1, "unable to switch fix_z and fix_y on together!"

        self.logger.info("Start Interpolation Analysis....")
        if fix_z:
            zs = torch.randn(nrow, 1, generator.z_dim, device=self.default_device)
            zs = zs.repeat(1, ncol, 1).view(-1, generator.z_dim)
            name = "fix_z"
        else:
            zs = interp(torch.randn(nrow, 1, generator.z_dim, device=self.default_device),
                        torch.randn(nrow, 1, generator.z_dim, device=self.default_device),
                        ncol - 2).view(-1, generator.z_dim)

        if fix_y:
            ys = sample_1hot(nrow, self.num_classes, device=self.default_device)
            ys = generator.shared(ys).view(nrow, 1, -1)
            ys = ys.repeat(1, ncol, 1).view(nrow * (ncol), -1)
            name = "fix_y"
        else:
            ys = interp(generator.shared(sample_1hot(nrow, self.num_classes)).view(nrow, 1, -1),
                        generator.shared(sample_1hot(nrow, self.num_classes)).view(nrow, 1, -1),
                        ncol-2).view(nrow * (ncol), -1)

        with torch.no_grad():
            interpolated_images = generator(zs, None, shared_label=ys, evaluation=True)

        plot_img_canvas((interpolated_images.detach().cpu()+1)/2, "./figures/{run_name}/Interpolated_images_{fix_flag}.png".\
                        format(run_name=self.run_name, fix_flag=name), self.logger, ncol)

        self.gen_model.train()
        ################################################################################################################################

    def linear_classification(self, total_step):
        toggle_grad(self.dis_model, False)
        self.dis_model.eval()
        self.linear_model.train()

        self.logger.info("Start training Linear classifier: {run_name}".format(run_name=self.run_name))
        train_iter = iter(self.train_dataloader)
        for step in tqdm(range(total_step)):
            self.L_optimizer.zero_grad()
            for acml_step in range(self.accumulation_steps):
                try:
                    if self.consistency_reg:
                        real_images, real_labels, real_images_aug = next(train_iter)
                    else:
                        real_images, real_labels = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_dataloader)
                    if self.consistency_reg:
                        real_images, real_labels, real_images_aug = next(train_iter)
                    else:
                        real_images, real_labels = next(train_iter)

                real_images, real_labels = real_images.to(self.default_device), real_labels.to(self.default_device)

                if self.conditional_strategy == "ContraGAN":
                    cls_proxies_real, cls_embed_real, dis_out_real = self.dis_model(real_images, real_labels, evaluation=True)
                elif self.conditional_strategy == "ACGAN":
                    cls_out_real, dis_out_real = self.dis_model(real_images, real_labels, evaluation=True)
                elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                    dis_out_real = self.dis_model(real_images, real_labels, evaluation=True)

                logits = self.linear_model(cls_embed_real)
                cls_loss = self.ce_loss(logits, real_labels)

                cls_loss.backward()

                self.L_optimizer.step()

        self.linear_model.eval()
        self.logger.info("Start evaluating Linear classifier: {run_name}".format(run_name=self.run_name))
        correct, total = 0, 0
        with torch.no_grad():
            eval_iter = iter(self.eval_dataloader)
            for i in range(len(self.eval_dataloader)):
                if self.consistency_reg:
                    real_images, real_labels, real_images_aug = next(eval_iter)
                else:
                    real_images, real_labels = next(eval_iter)

                real_images, real_labels = real_images.to(self.default_device), real_labels.to(self.default_device)

                if self.conditional_strategy == "ContraGAN":
                    cls_proxies_real, cls_embed_real, dis_out_real = self.dis_model(real_images, real_labels, evaluation=True)
                elif self.conditional_strategy == "ACGAN":
                    cls_out_real, dis_out_real = self.dis_model(real_images, real_labels, evaluation=True)
                elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                    dis_out_real = self.dis_model(real_images, real_labels, evaluation=True)

                logits_test = self.linear_model(cls_embed_real)
                _, prediction = torch.max(logits_test.data, 1)
                total += real_labels.size(0)
                correct += (prediction == real_labels).sum().item()

        self.logger.info("Accuracy of the network on the {total} {type} images: {acc}".\
                         format(total=total, type=self.type4eval_dataset, acc=100*correct/total))
