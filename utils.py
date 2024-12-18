from scipy import linalg
import torch
import torch.nn as nn
from networks import wresnet, resnet, lenet, conv3
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageFolder, STL10, USPS, SVHN


class BlackBox(nn.Module):
    def __init__(self, model:str, dataset:str, defense:bool=False, num_classes:int=10, window_size:int=64, device:str='cuda', threshold:float=float("inf"), save_dis:bool=False):
        super(BlackBox, self).__init__()
        self.device = device

        self.defense = defense
        self.window_size = window_size
        self.threshold = threshold

        self.model = get_model(model, dataset=dataset, is_vicitm=True, device=device).to(device)
        self.model.eval()

        self.detected = 0
        self.total = 1
        self.num_classes = num_classes
        self.save_dis = save_dis
        self.save_path = f"distance/{model}_{dataset}/"
        
        self.window = [[] for _ in range(self.num_classes)]
        
        if save_dis:
            self.dis_list = []

        if defense:
            self.mean = torch.load(f'mean&cov/{dataset}/{model}_mean_class.pt').to(self.device)
            self.cov = torch.load(f'mean&cov/{dataset}/{model}_cov_class.pt').to(self.device)

    def forward(self, x:torch.tensor) -> torch.tensor:
        with torch.no_grad():
            out, rep = self.model(x, return_rep=True)

            target = torch.argmax(out, dim=1)
            if self.defense:
                batch_size = rep.shape[0]
                self.total += batch_size

                self.window = [[] for _ in range(self.num_classes)]
                for i in range(batch_size):
                    self.window[target[i]].append(rep[i])
                
                dis_list = []
                total = 0
                for i in range(len(self.window)):
                    if len(self.window[i]) == 0:
                        continue
                    tensor = torch.stack(self.window[i])
                    mean = torch.mean(tensor, dim=0).to(self.device)
                    cov = torch.cov(tensor.T, correction=0).to(self.device)

                    d = calculate_fid_torch(self.mean[i], self.cov[i], mean, cov)

                    dis_list.append(d * len(self.window[i]))
                    total += len(self.window[i])
                dis = sum(dis_list)/total
                if self.save_dis:
                    self.dis_list.append(dis.cpu())


                if dis > self.threshold:
                    self.detected += batch_size
                    return torch.randn_like(out).to(self.device)

        return out

    def detected_ratio(self):
        if self.defense:
            return self.detected/self.total if self.total != 0 else 0
        else:
            return 0

    def restart(self):
        self.detected = 0
        self.total = 0
        self.window = [[] for _ in range(self.num_classes)]
        self.dis_list = []

    def save(self, dataset):
        data = np.array(self.dis_list)
        np.save(self.save_path+f'{dataset}_{self.window_size}.npy', data)



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            # print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=9e-1):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid_torch(mu1, sigma1, mu2, sigma2, eps=1e-6):

    diff = mu1 - mu2

    covmean = sqrtm(sigma1 @ sigma2)
    if not torch.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if torch.is_complex(covmean):
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)

    return (diff @ diff + torch.trace(sigma1)
            + torch.trace(sigma2) - 2 * tr_covmean)

def sqrtm(mat):
        if mat.size(0) != mat.size(1):
            raise ValueError("must be square matrix")
        eigvals, eigvecs = torch.linalg.eig(mat)
        eigvals_sqrt = torch.sqrt(eigvals)
        A_sqrt = eigvecs @ torch.diag(eigvals_sqrt) @ torch.linalg.inv(eigvecs)
        return A_sqrt


def get_model(arch:str, dataset:str=None, is_vicitm:bool=False, device='cuda', model_path='pretrain') -> nn.Module:
    if arch == 'conv3':
        net = conv3.Conv3(x_dim=32, n_channels=1)
        if is_vicitm:
            if dataset == 'mnist':
                net.load_state_dict(torch.load(
                    f'{model_path}/conv3_mnist.pth',
                    map_location=device), strict=True)
            elif dataset == 'fashion':
                net.load_state_dict(torch.load(
                    f'{model_path}/conv3_fashion.pth',
                    map_location=device), strict=True)
            else:
                raise NotImplementedError("no implement")

    elif arch == 'lenet':
        net = lenet.LeNet5()
        if is_vicitm:
            if dataset.lower() == 'mnist':
                net.load_state_dict(torch.load(
                    f'{model_path}/lenet_mnist.pth', 
                    map_location=device), strict=True)
            elif dataset.lower() == 'fashion':
                net.load_state_dict(torch.load(
                    f'{model_path}/lenet_fashion.pth', 
                    map_location=device), strict=True)
            else:
                raise NotImplementedError("no implement")

    elif arch == 'wrn16_4':
        net = wresnet.wrn_16_4(num_classes=10) if dataset == 'cifar10' else wresnet.wrn_16_4(num_classes=100)
        if is_vicitm:
            if dataset == 'cifar10':
                net.load_state_dict(torch.load(
                    f'{model_path}/wrn16_4_cifar10.pth',
                    map_location=device), strict=True)
            elif dataset == 'cifar100':
                net.load_state_dict(torch.load(
                    f'{model_path}/wrn16_4_cifar100.pth',
                    map_location=device), strict=True)
            else:
                raise NotImplementedError("no implement")

    elif arch == 'res18':
        net = resnet.resnet18(num_classes=17)
        if is_vicitm:
            if dataset == 'flower17':
                net.load_state_dict(torch.load(
                    f'{model_path}/res18_flower17.pth',
                    map_location=device),strict=True)
            else:
                raise NotImplementedError("no implement")
            
    return net


def get_dataloader(dataset, train=True, batch_size=64, augment=False, shuffle=True, drop_last=False, dataset_ID=None, filter=None, datasets_root='../datasets'):
    if dataset_ID == None:
        dataset_ID = dataset 
    transform = get_transform(dataset_ID, augment)

    if dataset.lower() == 'mnist':
        data = MNIST(datasets_root, train=train, transform=transform)

    elif dataset.lower() == 'fashion':
        data = FashionMNIST(datasets_root, train=train, transform=transform)

    elif dataset.lower() == 'cifar10':
        data = CIFAR10(datasets_root, train=train, transform=transform)

    elif dataset.lower() == 'cifar100':
        data = CIFAR100(datasets_root, train=train, transform=transform)

    elif dataset.lower() == 'flower17':
        t = 'train' if train else 'test'
        data = ImageFolder(
            root=f"{datasets_root}/flowers17/{t}/",
            transform=transform)

    elif dataset.lower() == 'indoor67':
        # t = 'train' if train else 'test'
        data = ImageFolder(
            root=f"{datasets_root}/indoors67/",
            transform=transform)

    elif dataset.lower() == 'tinyimagenet':
        t = 'train' if train else 'test'
        data = ImageFolder(
            root=f"{datasets_root}tiny-imagenet-200/{t}/",
            transform=transform)

    elif dataset.lower() == 'stl10':
        t = 'train' if train else 'test'
        data = STL10(datasets_root, split=t, transform=transform)
        if filter:
            data = FilteredDataset(data, filter, class_map=stl10_to_cifar10_map)

    elif dataset.lower() == 'usps':
        data = USPS(datasets_root, train=train, transform=transform)
        if filter:
            data = FilteredDataset(data, filter)

    elif dataset.lower() == 'svhn':
        t = 'train' if train else 'test'
        data = SVHN(datasets_root, split=t, transform=transform)
        if filter:
            data = FilteredDataset(data, filter)
    
    else:
        raise NotImplementedError("no implement")


    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=True)



def get_transform(dataset, augment=False):
    resize = transforms.Resize((32, 32)) if dataset.lower() in ['mnist', 'fashion', 'cifar10', 'cifar100'] else transforms.Compose([])

    if dataset == 'mnist':
        augment = transforms.Compose([])
    elif augment:
        augment = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]) if dataset in ['fashion', 'cifar10', 'cifar100'] \
            else transforms.Compose([transforms.RandomResizedCrop((224, 224)), transforms.RandomRotation(45), transforms.RandomHorizontalFlip(),])
    elif dataset == 'flower17':
        augment = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    else:
        augment = transforms.Compose([])

    if dataset.lower() == 'mnist':
        normalize = transforms.Normalize([0.1307], [0.3081])
    elif dataset.lower() == 'fashion':
        normalize = transforms.Normalize([0.2860], [0.3530])
    elif dataset.lower() == 'cifar10':
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    elif dataset.lower() == 'cifar100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    else:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return transforms.Compose([resize, augment, transforms.ToTensor(), normalize])



class FilteredDataset(Dataset):
    def __init__(self, dataset, filter_label, class_map=None):
        self.dataset = dataset
        self.filter_label = filter_label
        self.class_map = class_map
        self.filtered_indices = [i for i, (_, label) in enumerate(dataset) if label not in filter_label]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        data, label = self.dataset[original_idx]
        return data, label if self.class_map == None else self.class_map[label]
    

stl10_to_cifar10_map = {
    0: 0,  # airplane
    1: 2,  # bird
    2: 1,  # automobile
    3: 3,  # cat
    4: 4,  # deer
    5: 5,  # dog
    6: 7,  # horse
    8: 8,  # ship
    9: 9   # truck
} # 7 monkey is removed



def test(model, dataloader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input, target = batch[0].to(device), batch[1].to(device)
            total += input.shape[0]
            output = model(input)
            correct += torch.sum(torch.argmax(output, dim=1) == target)
    return (correct / total).item()