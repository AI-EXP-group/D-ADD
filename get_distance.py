import argparse
import numpy as np
from tqdm import tqdm
from utils import BlackBox, get_dataloader
from torch.utils.data import  Dataset
import torch

class FilteredDataset(Dataset):
    def __init__(self, dataset, filter_label):
        self.dataset = dataset
        self.filter_label = filter_label
        self.filtered_indices = [i for i, (_, label) in enumerate(dataset) if label not in filter_label]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        data, label = self.dataset[original_idx]
        return data, label



def main():
    parser = argparse.ArgumentParser(description="Parameters for calculating threshlod",)
    parser.add_argument('--model', default='wrn16_4', help='model(default: wrn16_4)',
                        choices=['lenet', 'conv3', 'wrn16_4', 'res18'])
    parser.add_argument('--dataset', default='cifar100', help='Dataset used to get distance(default: cifar10)',
                        choices=['mnist', 'fashion', 'cifar10', 'cifar100', 'flower17'])
    parser.add_argument('--train_set', default=False, help='whether use train set')
    parser.add_argument('--target_dataset', default='cifar10', help='Dataset of the model (default: cifar10)',
                        choices=['mnist', 'fashion', 'cifar10', 'cifar100', 'flower17'])
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes of target_dataset')
    parser.add_argument('--window_size', type=int, default=16, help='window size (mnist,fashion,cifar10:16, flower17:32, cifar100:256)')
    parser.add_argument('--data_path', default='../datasets', help='Path to store all the relevant datasetS.')
    parser.add_argument('--device', default='cuda:0', help='device to run')

    args = parser.parse_args()
    
    get_dis(args, dataset=args.dataset, target_dataset=args.target_dataset, model=args.model, window_size=args.window_size)



def get_dis(args, dataset, target_dataset, model, window_size):
    dataloader = get_dataloader(dataset, dataset_ID=target_dataset, train=args.train_set, batch_size=window_size, shuffle=True, drop_last=True)
    blackbox = BlackBox(model, target_dataset, defense=True, num_classes=args.num_classes, window_size=window_size, device=args.device, save_dis=True)
    
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, targets = batch[0].to(args.device), batch[1].to(args.device)
            output = blackbox(images)
            correct += torch.sum(torch.argmax(output, dim=1) == targets).item()
            total += images.shape[0]
    
    blackbox.save(dataset)
    
    if dataset == target_dataset:
        print("model acc:", round(correct / total, 4))
    print("max distance:", round(np.max(np.array(blackbox.dis_list)), 4))
    print("min distance:", round(np.min(np.array(blackbox.dis_list)), 4))
    print("average distance:", round(np.mean(np.array(blackbox.dis_list)), 4), "±", round(np.std(np.array(blackbox.dis_list)), 4))


if __name__ == '__main__':
    main()
