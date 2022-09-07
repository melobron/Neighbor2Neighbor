import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import json
import random
from tqdm import tqdm

from utils import *
from mask import generate_mask_pair, generate_subimages
from models.DnCNN import DnCNN
from dataset import ImageNetGray


class TrainNe2Ne:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.start_epoch = args.start_epoch
        self.decay_epoch = args.decay_epoch
        self.lr = args.lr
        self.noise = args.noise
        self.increase_ratio = args.increase_ratio
        self.rec_weight = args.rec_weight
        self.reg_weight = args.reg_weight

        # Loss
        self.criterion_mse = nn.MSELoss()

        # Transformation Parameters
        self.mean = args.mean
        self.std = args.std

        # Transform
        transform = transforms.Compose(get_transforms(args))

        # Models & Dataset
        self.train_dataset = ImageNetGray(noise=self.noise, train=True, transform=transform)
        self.test_dataset = ImageNetGray(noise=self.noise, train=False, transform=transform)
        self.model = DnCNN().to(self.device)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Scheduler
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(self.n_epochs, self.start_epoch, self.decay_epoch).step)

        # Directories
        self.exp_dir = make_exp_dir('./experiments/')['new_dir']
        self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path = os.path.join(self.exp_dir, 'results')

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def prepare(self):
        # Save Paths
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def train(self):
        print(self.device)
        self.prepare()

        for epoch in range(1, self.n_epochs + 1):
            with tqdm(self.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
                for batch, data in enumerate(tepoch):
                    self.model.train()
                    self.optimizer.zero_grad()

                    clean, noisy = data['clean'], data['noisy']
                    clean, noisy = clean.to(self.device), noisy.to(self.device)

                    mask1, mask2 = generate_mask_pair(noisy)
                    noisy_sub1 = generate_subimages(noisy, mask1)
                    noisy_sub2 = generate_subimages(noisy, mask2)
                    with torch.no_grad():
                        noisy_denoised = self.model(noisy)
                    noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
                    noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

                    noisy_output = self.model(noisy_sub1)
                    noisy_target = noisy_sub2
                    Lambda = epoch / self.n_epochs * self.increase_ratio
                    diff = noisy_output - noisy_target
                    exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

                    loss1 = torch.mean(diff**2)
                    loss2 = Lambda * torch.mean((diff - exp_diff)**2)
                    loss = self.rec_weight * loss1 + self.reg_weight * loss2

                    loss.backward()
                    self.optimizer.step()

                    tepoch.set_postfix(rec_loss=loss1.item(), reg_loss=loss2.item(),
                                       Lambda=Lambda, total_loss=loss.item())
                    self.summary.add_scalar('total_loss', loss.item(), epoch)
                    self.summary.add_scalar('rec_loss', loss1.item(), epoch)
                    self.summary.add_scalar('reg_loss', loss2.item(), epoch)
                    self.summary.add_scalar('Lambda', Lambda, epoch)

            self.scheduler.step()

            # Checkpoints
            if epoch % 10 == 0 or epoch == self.n_epochs:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, '{}epochs.pth'.format(epoch)))

            if epoch % 5 == 0:
                noisy_psnr, denoised_psnr = 0, 0
                noisy_ssim, denoised_ssim = 0, 0

                with torch.no_grad():
                    self.model.eval()

                    num_data = 10
                    for index in range(num_data):
                        data = self.test_dataset[index]
                        sample_clean, sample_noisy = data['clean'], data['noisy']
                        sample_noisy = torch.unsqueeze(sample_noisy, dim=0).to(self.device)

                        sample_denoised = self.model(sample_noisy)

                        if self.args.normalize:
                            sample_clean = denorm(sample_clean, mean=self.mean, std=self.std)
                            sample_noisy = denorm(sample_noisy, mean=self.mean, std=self.std)
                            sample_denoised = denorm(sample_denoised, mean=self.mean, std=self.std)

                        sample_clean, sample_noisy = tensor_to_numpy(sample_clean), tensor_to_numpy(sample_noisy)
                        sample_denoised = tensor_to_numpy(sample_denoised)

                        sample_clean, sample_noisy = np.squeeze(sample_clean), np.squeeze(sample_noisy)
                        sample_denoised = np.squeeze(sample_denoised)

                        # Calculate PSNR
                        n_psnr = psnr(sample_clean, sample_noisy, data_range=1)
                        d_psnr = psnr(sample_clean, sample_denoised, data_range=1)
                        # print('{}th image PSNR | noisy:{:.3f}, denoised:{:.3f}'.format(index + 1, n_psnr, d_psnr))

                        noisy_psnr += n_psnr / num_data
                        denoised_psnr += d_psnr / num_data

                        # Calculate SSIM
                        n_ssim = ssim(sample_clean, sample_noisy, data_range=1)
                        d_ssim = ssim(sample_clean, sample_denoised, data_range=1)
                        # print('{}th image SSIM | noisy:{:.3f}, denoised:{:.3f}'.format(index + 1, n_ssim, d_ssim))

                        noisy_ssim += n_ssim / num_data
                        denoised_ssim += d_ssim / num_data

                        # Save sample image
                        sample_clean, sample_noisy = 255. * np.clip(sample_clean, 0., 1.), 255. * np.clip(sample_noisy, 0., 1.)
                        sample_denoised = 255. * np.clip(sample_denoised, 0., 1.)

                        if index == 0:
                            cv2.imwrite(os.path.join(self.result_path, 'clean_{}epochs.png'.format(epoch)), sample_clean)
                            cv2.imwrite(os.path.join(self.result_path, 'noisy_{}epochs.png'.format(epoch)), sample_noisy)
                            cv2.imwrite(os.path.join(self.result_path, 'simple_output_{}epochs.png'.format(epoch)), sample_denoised)

                    # PSNR, SSIM
                    print('Average PSNR | noisy:{:.3f}, denoised:{:.3f}'.format(noisy_psnr, denoised_psnr))
                    print('Average SSIM | noisy:{:.3f}, denoised:{:.3f}'.format(noisy_ssim, denoised_ssim))
                    self.summary.add_scalar('avg_denoised_psnr', denoised_psnr, epoch)
                    self.summary.add_scalar('avg_denoised_ssim', denoised_ssim, epoch)

        self.summary.close()










