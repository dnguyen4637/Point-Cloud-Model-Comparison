"""
Usage:
python main.py --model PointMLP --msg demo
"""
import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import sklearn.metrics as metrics
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from data import ModelNet40
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointNet', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.005, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    device = 'cuda'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # checkpoint path
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    message = time_str if args.msg is None else "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message + '-' + str(args.seed)
    mkdir_p(args.checkpoint)

    # Logging
    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(x):
        screen_logger.info(x)
        print(x)

    printf(f"args: {args}")
    printf('==> Building model..')
    net = models.__dict__[args.model]()
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    criterion = cal_loss

    # Logging setup
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
    logger.set_names([
        "Epoch", "LR",
        "Train-Loss", "Train-Acc-B", "Train-Acc", "Train-Precision", "Train-Recall", "Train-F1",
        "Valid-Loss", "Valid-Acc-B", "Valid-Acc", "Valid-Precision", "Valid-Recall", "Valid-F1"
    ])

    printf('==> Preparing data..')
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points),
                              num_workers=args.workers, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             num_workers=args.workers, batch_size=args.batch_size // 2,
                             shuffle=False, drop_last=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate,
                                momentum=0.9, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr)

    best_test_acc = 0.

    for epoch in range(args.epoch):
        printf(f"Epoch({epoch+1}/{args.epoch}) Learning Rate {optimizer.param_groups[0]['lr']}:")

        train_out = train(net, train_loader, optimizer, criterion, device)
        test_out = validate(net, test_loader, criterion, device)
        scheduler.step()

        # Save best model
        is_best = test_out["acc"] > best_test_acc
        best_test_acc = max(best_test_acc, test_out["acc"])

        save_model(
            net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc, best_train_acc=train_out["acc"],
            best_test_acc_avg=test_out["acc_avg"], best_train_acc_avg=train_out["acc_avg"],
            best_test_loss=test_out["loss"], best_train_loss=train_out["loss"],
            optimizer=optimizer.state_dict()
        )

        # Log ALL metrics
        logger.append([
            epoch,
            optimizer.param_groups[0]['lr'],

            train_out["loss"], train_out["acc_avg"], train_out["acc"],
            train_out["precision"], train_out["recall"], train_out["f1"],

            test_out["loss"], test_out["acc_avg"], test_out["acc"],
            test_out["precision"], test_out["recall"], test_out["f1"]
        ])

        # Terminal output
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% "
            f"acc:{train_out['acc']}% Precision:{train_out['precision']} "
            f"Recall:{train_out['recall']} F1:{train_out['f1']} time:{train_out['time']}s"
        )
        printf(
            f"Testing  loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% Precision:{test_out['precision']} "
            f"Recall:{test_out['recall']} F1:{test_out['f1']} time:{test_out['time']}s "
            f"[best test acc: {best_test_acc}%]\n"
        )

    logger.close()
    printf("Training Complete.")


def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss, correct, total = 0, 0, 0
    train_true, train_pred = [], []
    start = datetime.datetime.now()

    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)

        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        train_loss += loss.item()
        preds = logits.max(1)[1]
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     f"Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})")

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)

    precision = metrics.precision_score(train_true, train_pred, average='macro', zero_division=0)
    recall = metrics.recall_score(train_true, train_pred, average='macro', zero_division=0)
    f1 = metrics.f1_score(train_true, train_pred, average='macro', zero_division=0)

    return {
        "loss": float(f"{train_loss/(batch_idx+1):.3f}"),
        "acc": float(f"{100 * metrics.accuracy_score(train_true, train_pred):.3f}"),
        "acc_avg": float(f"{100 * metrics.balanced_accuracy_score(train_true, train_pred):.3f}"),
        "precision": float(f"{100 * precision:.3f}"),
        "recall": float(f"{100 * recall:.3f}"),
        "f1": float(f"{100 * f1:.3f}"),
        "time": int((datetime.datetime.now() - start).total_seconds())
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    test_true, test_pred = [], []
    start = datetime.datetime.now()

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)

            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()

            preds = logits.max(1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.cpu().numpy())

            total += label.size(0)
            correct += preds.eq(label).sum().item()

            progress_bar(batch_idx, len(testloader),
                         f"Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})")

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    precision = metrics.precision_score(test_true, test_pred, average='macro', zero_division=0)
    recall = metrics.recall_score(test_true, test_pred, average='macro', zero_division=0)
    f1 = metrics.f1_score(test_true, test_pred, average='macro', zero_division=0)

    return {
        "loss": float(f"{test_loss/(batch_idx+1):.3f}"),
        "acc": float(f"{100 * metrics.accuracy_score(test_true, test_pred):.3f}"),
        "acc_avg": float(f"{100 * metrics.balanced_accuracy_score(test_true, test_pred):.3f}"),
        "precision": float(f"{100 * precision:.3f}"),
        "recall": float(f"{100 * recall:.3f}"),
        "f1": float(f"{100 * f1:.3f}"),
        "time": int((datetime.datetime.now() - start).total_seconds())
    }


if __name__ == '__main__':
    main()
