import argparse
import torch
from tqdm import tqdm
from utils import get_model, get_dataloader, test



def main():
    parser = argparse.ArgumentParser(description="Parameters for calculating threshlod",)
    parser.add_argument('--model', default='wrn16_4', help='model used to train (default: wrn16_4)',
                        choices=['lenet', 'conv3', 'wrn16_4', 'res18'])
    parser.add_argument('--dataset', default='cifar10', help='Dataset used to train a model (default: cifar10)',
                        choices=['mnist', 'fashion', 'cifar10', 'cifar100', 'flower17'])
    parser.add_argument('--num_classes', default=10, type=int, help='num_classes of dataset')
    parser.add_argument('--data_path', default='../datasets', help='Path to store all the relevant datasetS.')
    parser.add_argument('--model_path', default='pretrain', help='Path to store all the relevant datasetS.')
    parser.add_argument('--device', default='cuda:0', help='device to run')
    args = parser.parse_args()
    
    get_main_cov(args, args.dataset, args.model)


def get_main_cov(args, dataset, model):
    model = get_model(model, dataset, is_vicitm=True, device=args.device, model_path=args.model_path)
    model.eval()
    model.to(args.device)
    print("model acc:", test(model, get_dataloader(dataset, train=False), args.device))
    dataloader = get_dataloader(dataset, train=True, batch_size=100, augment=False, shuffle=False, datasets_root=args.data_path)
    
    
    with torch.no_grad():
        feature_list = [[] for _ in range(args.num_classes)]
        for batch in tqdm(dataloader):
            images, targets = batch[0].to(args.device), batch[1]
            _, features = model(images, return_rep=True)
            for i in range(features.shape[0]):
                feature_list[targets[i]].append(features[i])

    feature_class = [torch.stack(feature_list[i]) for i in range(args.num_classes)] # [num_classes, num_features, dim]

    mean_class = [torch.mean(feature_class[i], dim=0) for i in range(len(feature_class))] # [num_classes, dim]
    cov_class = [torch.cov(feature_class[i].T, correction=0) for i in range(len(feature_class))] # [num_classes, dim, dim]
    
    torch.save(torch.stack(mean_class), f'mean&cov/{args.dataset}/{args.model}_mean_class.pt')
    torch.save(torch.stack(cov_class), f'mean&cov/{args.dataset}/{args.model}_cov_class.pt')
    
    
if __name__ == '__main__':
    main()