import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import *
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Parameters for calculating threshlod",)
    parser.add_argument('--victim_model', default='wrn16_4', help='victim model',
                        choices=['lenet', 'conv3', 'wrn16_4', 'res18'])
    parser.add_argument('--surrogate_model', default='wrn16_4', help='surrogate model',
                        choices=['lenet', 'conv3', 'wrn16_4', 'res18'])
    parser.add_argument('--target_dataset', default='cifar10', help='Dataset of the model',
                        choices=['mnist', 'fashion', 'cifar10', 'cifar100', 'flower17'])
    parser.add_argument('--surrogate_dataset', default='cifar100', help='Dataset used to get distance',
                        choices=['mnist', 'fashion', 'cifar10', 'cifar100', 'flower17'])
    parser.add_argument('--train_set', default=True, help='whether use train set')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes of target_dataset')
    
    parser.add_argument('--window_size', default=16, type=int, 
                        help='window size (mnist,fashion,cifar10:16, flower17:32, cifar100:256)')
    parser.add_argument('--defense', default=True, type=float, help='use D-ADD defense')
    # different model weights may have slightly different threshold
    parser.add_argument('--threshold', default=11.6, type=float,
                        help='distance threshold for D-ADD defense (2247, 650, 11.6, 82.37, 14.85 for MNIST, FashionMNIST, CIFAR10, CIFAR100, Flower17)')
    parser.add_argument('--query_budget', default=50000, type=int, 
                        help='MNIST, FahionMNIST:60000, CIFAR10, CIFAR100:50000, Flower17:15620')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate for surrogate model')
    parser.add_argument('--e', default=30, type=int, help='epoch')
    parser.add_argument('--data_path', default='../datasets', help='Path to store all the relevant datasetS.')
    parser.add_argument('--device', default='cuda:0', help='device to run')

    args = parser.parse_args()


    victim_model = BlackBox(args.victim_model, args.target_dataset, defense=args.defense, num_classes=args.num_classes, window_size=args.window_size, device=args.device, threshold=args.threshold)
    surrogate_model = get_model(args.surrogate_model, dataset=args.target_dataset, device=args.device)
    
    dataloader_sur = get_dataloader(args.surrogate_dataset, dataset_ID=args.target_dataset, train=args.train_set, batch_size=args.window_size, shuffle=True, drop_last=True)
    dataloader_test = get_dataloader(args.target_dataset, train=False, batch_size=args.window_size, shuffle=False)

    print("Victim model acc:", round(test(victim_model, dataloader_test, args.device), 4))

    knockoff(victim_model, surrogate_model, dataloader_sur, dataloader_test, args.query_budget, args.e, args.lr, args.device)
    print('Surrogate model acc:', round(test(surrogate_model, dataloader_test, args.device), 4))



def loss_fn_kd(outputs, labels, teacher_outputs, T=3, alpha=0.3):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    CE_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / T, dim=1),
                                                  F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + CE_loss
    return KD_loss


def transfer(teacher_model, dataloader_sur, query_budget, device):
    xs = torch.tensor([])
    ys = torch.tensor([])
    queries = 0
    teacher_model.eval()
    with torch.no_grad():
        for epoch in range(100):
            for batch in tqdm(dataloader_sur):
                x = batch[0].to(device)

                y = teacher_model(x)

                xs = torch.cat((xs, x.cpu()), dim=0)
                ys = torch.cat((ys, y.cpu()), dim=0)

                queries += x.shape[0]
                if queries >= query_budget:
                    break
            if queries >= query_budget:
                break
    return TensorDataset(xs, ys)


def train(student_model, dataloader, dataloader_test, epoch, lr, device):
    criterion = loss_fn_kd
    opt = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch, last_epoch=-1)

    best_acc = 0

    for i in range(epoch):
        print(f'epoch {i + 1}')
        student_model.train()
        loss_list = []
        for batch in dataloader:
            input, target = batch[0].to(device), batch[1].to(device)

            output = student_model(input)
            hard = target.data.max(1)[1]
            loss = criterion(output, hard, target)
            loss_list.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        acc = test(student_model, dataloader_test, device)
        if acc > best_acc:
            best_acc = acc
        print('loss: ', round(sum(loss_list) / len(loss_list), 5))
        print('student acc: ', round(acc, 5), 'best acc: ', round(best_acc, 5))

    return best_acc


def knockoff(teacher_model, student_model, dataloader_sur, dataloader_test, query_budget, epoch, lr,
             device):
    print('transfer')
    dataset_sur = transfer(teacher_model, dataloader_sur, query_budget, device)
    print("detected ratio: ", teacher_model.detected_ratio())
    print('query budget: ', len(dataset_sur))
    dataloader = DataLoader(dataset_sur, batch_size=64, shuffle=True)
    print('train')
    acc = train(student_model, dataloader, dataloader_test, epoch, lr, device)
    return acc


if __name__ == '__main__':
    main()
