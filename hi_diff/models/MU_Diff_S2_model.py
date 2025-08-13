import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import os
from os.path import join as opj

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from hi_diff.utils.base_model import BaseModel
from torch.nn import functional as F
from functools import partial
import numpy as np
from hi_diff.utils.beta_schedule import make_beta_schedule, default
from ldm.ddpm import DDPM
from hi_diff.utils.ce_model import CEBlurNet
from hi_diff.utils.utils_image_kair import tensor2uint, imsave


@MODEL_REGISTRY.register()
class MU_Diff_S2(BaseModel):
    """MU-Diff model for test."""

    def __init__(self, opt):
        super(MU_Diff_S2, self).__init__(opt)

        # define network
        self.net_le = build_network(opt['network_le'])
        self.net_le = self.model_to_device(self.net_le)
        self.print_network(self.net_le)

        self.net_le_dm = build_network(opt['network_le_dm'])
        self.net_le_dm = self.model_to_device(self.net_le_dm)
        self.print_network(self.net_le_dm)

        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.frame_n = opt['ce_net']['frame_n']
        self.ce_code_n = opt['ce_net']['ce_code_n']

        self.opt_cenet = opt['ce_net']
        self.BlurNet = CEBlurNet(
            sigma_range=self.opt_cenet['sigma_range'], 
            test_sigma_range=self.opt_cenet['test_sigma_range'],
            ce_code_n=self.opt_cenet['ce_code_n'],
            frame_n=self.opt_cenet['frame_n'],
            ce_code_init=self.opt_cenet['ce_code_init'],
            opt_cecode=self.opt_cenet['opt_cecode'],
            binary_fc=self.opt_cenet['binary_fc']
        )

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_le', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_le', 'params')
            self.load_network(self.net_le, load_path, self.opt['path'].get('strict_load_le', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_le_dm', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_le_dm', 'params')
            self.load_network(self.net_le_dm, load_path, self.opt['path'].get('strict_load_le_dm', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # diffusion
        self.apply_ldm = self.opt['diffusion_schedule'].get('apply_ldm', None)
        if self.apply_ldm:
            # apply LDM implementation
            self.diffusion = DDPM(denoise=self.net_d, 
                                  condition=self.net_le_dm, 
                                  n_feats=opt['network_g']['embed_dim'], 
                                  group=opt['network_g']['group'],
                                  linear_start= self.opt['diffusion_schedule']['linear_start'],
                                  linear_end= self.opt['diffusion_schedule']['linear_end'], 
                                  timesteps = self.opt['diffusion_schedule']['timesteps'])
            self.diffusion = self.model_to_device(self.diffusion)
        else:
            # implemented locally
            self.set_new_noise_schedule(self.opt['diffusion_schedule'], self.device)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.net_d.train()
        self.net_le.train()
        self.net_le_dm.train()
        if self.apply_ldm:
            self.diffusion.train()
        
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            print("TODO")

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pix_diff = build_loss(train_opt['pixel_diff_opt']).to(self.device)
        else:
            self.cri_pix = None
            self.cri_pix_diff = None

        if train_opt.get('pixel_opt_reblur'):
            self.cri_reblur = build_loss(train_opt['pixel_opt_reblur']).to(self.device)
        else:
            self.cri_reblur = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Network G: Params {k} will not be optimized.')

        if self.apply_ldm:
            for k, v in self.diffusion.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Network Diffusion: Params {k} will not be optimized.')
        else:
            for k, v in self.net_le_dm.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Network LE-DM: Params {k} will not be optimized.')

            for k, v in self.net_d.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Network D: Params {k} will not be optimized.')

        optim_type = train_opt['optim_total'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_total = torch.optim.Adam(optim_params, **train_opt['optim_total'])
        elif optim_type == 'AdamW':
            self.optimizer_total = torch.optim.AdamW(optim_params, **train_opt['optim_total'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_total)

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # β1, β2, ..., βΤ (T)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['timesteps'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        # α1, α2, ..., αΤ (T)
        alphas = 1. - betas
        # α1, α1α2, ..., α1α2...αΤ (T)
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # 1, α1, α1α2, ...., α1α2...αΤ-1 (T)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # 1, √α1, √α1α2, ...., √α1α2...αΤ (T+1)
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        if condition_x is None:
            raise RuntimeError('Must have LQ/LR condition')

        if ema_model:
            print("TODO")
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.net_d(x, condition_x, torch.full(x.shape, t+1, device=self.betas.device, dtype=torch.long)))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance
    
    def p_sample_wo_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean
    
    def p_sample_loop_wo_variance(self, x_in, x_noisy, ema_model=False):
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample_wo_variance(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def p_sample(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean

    def p_sample_loop(self, x_in, x_noisy, ema_model=False):
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def q_sample(self, x_start, sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            sqrt_alpha_cumprod * x_start +
            (1 - sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def feed_data(self, data):
        # self.lq = data['lq'].to(self.device)
        # if 'gt' in data:
        #     self.gt = data['gt'].to(self.device)
        vid = data.to(self.device).float() / 255

        interp_scale = self.frame_n // self.ce_code_n
        self.gt = vid[:, ::interp_scale]

        ce_blur_img_noisy, time_idx, ce_code_up, ce_blur_img = self.BlurNet(vid)
        self.lq = ce_blur_img_noisy

        self.ce_code_up = ce_code_up.to(self.device)
        self.ce_blur_img = ce_blur_img
        self.time_idx = time_idx

    def optimize_parameters(self, current_iter, noise=None):
        # freeze c1 (cpen_s1)
        for p in self.net_le.parameters():
            p.requires_grad = False
        
        self.optimizer_total.zero_grad()
        # prior_z = self.net_le(self.lq, self.gt)
        prior_z_list = self.net_le(ce_blur=self.lq, time_idx=self.time_idx, ce_code=self.ce_code_up, gt=self.gt) # (8, b, 64, 128)
        prior_list = []

        if self.apply_ldm:
            # prior, _=self.diffusion(self.lq, prior_z)
            for i in range(self.frame_n):
                prior_z = prior_z_list[i]
                prior, _ = self.diffusion(self.lq, prior_z)
                prior_list.append(prior)
        else:
            # prior_d = self.net_le_dm(self.lq)
            prior_d_list = self.net_le_dm(ce_blur=self.lq, time_idx=self.time_idx, ce_code=self.ce_code_up) # (8, b, 64, 128)
            # diffusion-forward
            t = self.opt['diffusion_schedule']['timesteps']
            # # [b, 4c']
            # noise = default(noise, lambda: torch.randn_like(prior_z))
            # # sample xt/x_noisy (from x0/x_start)
            # prior_noisy = self.q_sample(
            #     x_start=prior_z, sqrt_alpha_cumprod=self.alphas_cumprod[t-1],
            #     noise=noise)
            # # diffusion-reverse
            # prior = self.p_sample_loop_wo_variance(prior_d, prior_noisy)
            for i in range(self.frame_n):
                prior_z = prior_z_list[i]
                prior_d = prior_d_list[i]
                noise = default(noise, lambda: torch.randn_like(prior_z))
                prior_noisy = self.q_sample(
                    x_start=prior_z, sqrt_alpha_cumprod=self.alphas_cumprod[t-1],
                    noise=noise)
                prior = self.p_sample_loop_wo_variance(prior_d, prior_noisy) # (b, 64, 128)
                prior_list.append(prior)

        # ir
        # self.output = self.net_g(self.lq, prior)
        output_list = []
        for i in range(self.frame_n):
            output = self.net_g(self.lq, prior_list[i])
            output_list.append(output)
        self.output = torch.stack(output_list, dim=1)

        output_ = torch.flatten(self.output, end_dim=1)
        target_ = torch.flatten(self.gt, end_dim=1)

        prior_ = torch.flatten(torch.stack(prior_list, dim=1), end_dim=1)
        prior_z_ = torch.flatten(torch.stack(prior_z_list, dim=1), end_dim=1)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(output_, target_)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_pix_diff:
            l_pix_diff = self.cri_pix_diff(prior_, prior_z_)
            l_total += l_pix_diff
            loss_dict['l_pix_diff'] = l_pix_diff

        # reblur loss
        if self.cri_reblur:
            _, _, _, ce_output = self.BlurNet(self.output)
            l_reblur = self.cri_reblur(ce_output, self.ce_blur_img)
            l_total += l_reblur
            loss_dict['l_reblur'] = l_reblur

        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            if self.apply_ldm:
                torch.nn.utils.clip_grad_norm_(list(self.net_g.parameters()) + list(self.diffusion.parameters()), 0.01)
            else:
                torch.nn.utils.clip_grad_norm_(list(self.net_g.parameters()) + list(self.net_le_dm.parameters()) + list(self.net_d.parameters()), 0.01)
        self.optimizer_total.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        scale = self.opt.get('scale', 1)
        window_size = 8
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        if hasattr(self, 'net_g_ema'):
            print("TODO")
        else:
            if self.apply_ldm:
                self.net_g.eval()
                self.diffusion.eval()

                with torch.no_grad():
                    prior = self.diffusion(img)
                    self.output = self.net_g(img, prior)
                    self.diffusion.train()

                self.net_g.train()
                self.diffusion.train()
            else:
                self.net_le.eval()
                self.net_le_dm.eval()
                self.net_d.eval()
                self.net_g.eval()

                with torch.no_grad():
                    # prior_c = self.net_le_dm(img)
                    # prior_noisy = torch.randn_like(prior_c)
                    # prior = self.p_sample_loop(prior_c, prior_noisy)
                    # self.output = self.net_g(img, prior)
                    prior_c_list = self.net_le_dm(ce_blur=img, time_idx=self.time_idx, ce_code=self.ce_code_up)
                    output_list = []
                    for i in range(self.frame_n):
                        prior_c = prior_c_list[i]
                        prior_noisy = torch.randn_like(prior_c)
                        prior = self.p_sample_loop(prior_c, prior_noisy)
                        output = self.net_g(img, prior)
                        output_list.append(output)
                    self.output = torch.stack(output_list, dim=1)
                    self.output = torch.clamp(self.output, 0, 1)

                self.net_le.train()
                self.net_le_dm.train()
                self.net_d.train()
                self.net_g.train()

        _, _, _, h, w = self.output.size()
        self.output = self.output[:, :, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # dataset_name = dataloader.dataset.opt['name']
        dataset_name = 'ValSet'
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_datas = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            img_name = 'val_img'
            N, F, C, Hx, Wx = val_data.shape
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            # sr_img = tensor2img([visuals['result']])
            # metric_data['img'] = sr_img
            # if 'gt' in visuals:
            #     gt_img = tensor2img([visuals['gt']])
            #     metric_data['img2'] = gt_img
            #     del self.gt

            sr_imgs = tensor2img(list(torch.split(visuals['result'], 1, dim=0)))
            metric_datas['img'] = sr_imgs
            if 'gt' in visuals:
                gt_imgs = tensor2img(list(torch.split(visuals['gt'], 1, dim=0)))
                metric_datas['img2'] = gt_imgs

            # tentative for out of GPU memory
            # del self.lq
            # del self.output
            # torch.cuda.empty_cache()

            # TODO: save img
            ce_code = self.opt_cenet['ce_code_init']
            if save_img:
                scale_fc = len(ce_code)/sum(ce_code)
                if not osp.exists(opj(self.opt['path']['visualization'], 'input')):
                    for sub_dir in ['input', 'output', 'target']:
                        os.makedirs(opj(self.opt['path']['visualization'], sub_dir))
                for k, (in_img, out_img, gt_img) in enumerate(zip(self.lq, self.output, self.gt)):
                    in_img = tensor2uint(in_img*scale_fc)
                    imsave(
                        in_img, opj(self.opt['path']['visualization'], 'input', f'in-frame#{idx*N+k+1:04d}.png'))
                    for j in range(len(ce_code)):
                        out_img_j = tensor2uint(out_img[j])
                        gt_img_j = tensor2uint(gt_img[j])
                        imsave(
                            out_img_j, opj(self.opt['path']['visualization'], 'output', f'out-frame#{idx*N+k+1:04d}-{j+1:04d}.png'))
                        imsave(
                            gt_img_j, opj(self.opt['path']['visualization'], 'target', f'gt-frame#{idx*N+k+1:04d}-{j+1:04d}.png'))
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    # self.metric_results[name] += calculate_metric(metric_data, opt_)
                    total_metric = np.zeros(len(metric_datas['img']))
                    metric_data = dict()
                    for k, (out_img, gt_img) in enumerate(zip(metric_datas['img'], metric_datas['img2'])):
                        metric_data['img'] = out_img
                        metric_data['img2'] = gt_img
                        total_metric[k] = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += np.mean(total_metric)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # out_dict['result'] = self.output.detach().cpu()
        out_dict['result'] = torch.flatten(self.output, end_dim=1).detach().cpu()
        if hasattr(self, 'gt'):
            # out_dict['gt'] = self.gt.detach().cpu()
            out_dict['gt'] = torch.flatten(self.gt, end_dim=1).detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            print("TODO")
        else:
            if self.apply_ldm:
                if self.opt['dist']:
                    self.net_le_dm = self.diffusion.module.condition
                    self.net_d = self.diffusion.module.model
                else:
                    self.net_le_dm = self.diffusion.condition
                    self.net_d = self.diffusion.model
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.net_le_dm, 'net_le_dm', current_iter)
            # self.save_network(self.net_le, 'net_le', current_iter)
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
