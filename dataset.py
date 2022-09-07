from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import *


class SEM(Dataset):
    def __init__(self, dataset='SEM1', train=True, transform=None):
        super(SEM, self).__init__()

        self.train = train
        data_dir = os.path.join('../all_datasets/', dataset)
        if train:
            self.noisy_dir = os.path.join(data_dir, 'train')
            self.noisy_paths = sorted(make_dataset(self.noisy_dir))
        else:
            self.noisy_dir = os.path.join(data_dir, 'test')
            self.clean_dir = os.path.join(data_dir, 'test_gt')
            self.noisy_paths = sorted(make_dataset(self.noisy_dir))
            self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        if self.train:
            noisy_path = self.noisy_paths[index]
            noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE) / 255.
            noisy = self.transform(noisy)
            noisy = noisy.type(torch.FloatTensor)
            return noisy
        else:
            clean_path = self.clean_paths[index]
            noisy_path = self.noisy_paths[index]
            clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE) / 255.
            noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE) / 255.
            clean, noisy = self.transform(clean), self.transform(noisy)
            clean, noisy = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor)
            return {'noisy': noisy, 'clean': clean}

    def __len__(self):
        return len(self.noisy_paths)


class ImageNetGray(Dataset):
    def __init__(self, data_dir='../all_datasets/ImageNet_1000_Gray/', noise='gauss_25', train=True, transform=None):
        super(ImageNetGray, self).__init__()

        self.noise_type, self.noise_intensity = noise.split('_')[0], float(noise.split('_')[1]) / 255.

        if train:
            self.clean_dir = os.path.join(data_dir, 'train')
        else:
            self.clean_dir = os.path.join(data_dir, 'test')

        self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE) / 255.
        if self.noise_type == 'gauss':
            noisy = clean + np.random.randn(*clean.shape) * self.noise_intensity
        elif self.noise_type == 'poisson':
            noisy = np.random.poisson(clean * 255. * self.noise_intensity) / self.noise_intensity / 255.
        else:
            raise NotImplementedError('wrong type of noise')
        clean, noisy = self.transform(clean), self.transform(noisy)
        clean, noisy = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor)
        return {'clean': clean, 'noisy': noisy}

    def __len__(self):
        return len(self.clean_paths)










