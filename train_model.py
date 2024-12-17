import argparse
import torch
from tqdm import tqdm
from utils import get_model, get_dataloader, test
    


def main():
    parser = argparse.ArgumentParser(description="Parameters for training model",)
    parser.add_argument('--model', default='wrn16_4', help='model used to train (default: wrn16_4)',
                        choices=['lenet', 'conv3', 'wrn16_4', 'res18'])
    parser.add_argument('--dataset', default='cifar10', help='Dataset used to train a model (default: cifar10)',
                        choices=['mnist', 'fashion', 'cifar10', 'cifar100', 'flower17'])
    parser.add_argument('--epoch', type=int, default=50, help='epoch (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (mnist, fashion:0.01, others:0.1)')
    parser.add_argument('--save_path', default='pretrain/wrn16_4_cifar10.pth', help='Path to save model parameter.')
    parser.add_argument('--data_path', default='../datasets', help='Path to store all the relevant datasetS.')
    parser.add_argument('--device', default='cuda:0', help='device to run')

    args = parser.parse_args()
    
    train(args, args.model, args.dataset, args.lr, args.epoch, args.batch_size)


def train(args, model='wrn16_4', dataset='cifar10', lr=0.1, epochs=50, batch_size=128):
    
    dl_train = get_dataloader(args.dataset, train=True, batch_size=batch_size, augment=True, shuffle=True, datasets_root=args.data_path)
    dl_test = get_dataloader(args.dataset, train=False, batch_size=batch_size, augment=False, shuffle=True, datasets_root=args.data_path)


    model = get_model(model, dataset, False, device=args.device)
    model.to(args.device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), ncols=100):
        model.train()
        for batch in dl_train:
            input, target = batch[0].to(args.device), batch[1].to(args.device)
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        if (epoch+1) % 10 == 0:
            print()
            print('epoch:', epoch+1)
            acc = round(test(model, dl_test, args.device), 4)
            print('ACC:', acc)

    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    main()
