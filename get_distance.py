import argparse
import numpy as np
from tqdm import tqdm
from utils import BlackBox, get_dataloader
import ast
import torch

def str_to_list(string):
    """
    string: "[1, 2, 3]" -> list: [1, 2, 3]
    """
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list format: {string}")


def main():
    parser = argparse.ArgumentParser(description="Parameters for calculating threshlod",)
    parser.add_argument('--model', default='wrn16_4', help='model(default: wrn16_4)',
                        choices=['lenet', 'conv3', 'wrn16_4', 'res18'])
    parser.add_argument('--dataset', default='cifar100', help='Dataset used to get distance(default: cifar10)',
                        choices=['mnist', 'fashion', 'cifar10', 'cifar100', 'flower17', 'stl10', 'usps', 'indoor67'])
    parser.add_argument('--data_filter', default=None, type=str_to_list, help='get dataset without these labels (example: [1, 2, 3]).\
        stl10 for cifar10 must filter label 7')
    parser.add_argument('--train_set', default=True, help='use train set or test set')
    parser.add_argument('--target_dataset', default='cifar10', help='Dataset of the model',
                        choices=['mnist', 'fashion', 'cifar10', 'cifar100', 'flower17'])
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes of target_dataset')
    parser.add_argument('--defense', default=True, type=bool, help='use D-ADD defense')
    parser.add_argument('--threshold', default=11.6, type=float,
                        help='distance threshold for D-ADD defense (2247, 650, 11.6, 82.37, 14.85 for MNIST, FashionMNIST, CIFAR10, CIFAR100, Flower17)')
    parser.add_argument('--window_size', default=16, type=int, help='window size (mnist,fashion,cifar10:16, flower17:32, cifar100:256)')
    parser.add_argument('--data_path', default='../datasets', help='Path to store all the relevant datasetS.')
    parser.add_argument('--device', default='cuda:0', help='device to run')

    args = parser.parse_args()
    
    get_dis(args, dataset=args.dataset, target_dataset=args.target_dataset, model=args.model, window_size=args.window_size)



def get_dis(args, dataset, target_dataset, model, window_size):
    dataloader = get_dataloader(dataset, dataset_ID=target_dataset, train=args.train_set, batch_size=window_size, filter=args.data_filter, shuffle=True, drop_last=True, datasets_root=args.data_path)
    blackbox = BlackBox(model, target_dataset, defense=args.defense, num_classes=args.num_classes, window_size=window_size, device=args.device, threshold=args.threshold, save_dis=True)
    
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, targets = batch[0].to(args.device), batch[1].to(args.device)
            output = blackbox(images)
            correct += torch.sum(torch.argmax(output, dim=1) == targets).item()
            total += images.shape[0]
    
    blackbox.save(dataset)
    
    print("model acc:", round(correct / total, 4))
    if args.defense:
        print("max distance:", round(np.max(np.array(blackbox.dis_list)), 4))
        print("min distance:", round(np.min(np.array(blackbox.dis_list)), 4))
        print("average distance:", round(np.mean(np.array(blackbox.dis_list)), 4), "±", round(np.std(np.array(blackbox.dis_list)), 4))
        print("detected malicious rate:", round(blackbox.detected_ratio(), 4))


if __name__ == '__main__':
    main()
