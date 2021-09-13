from matplotlib import pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import confusion_matrix

from helpers import Dataset as Dataset
from models import Optim as Optim
from models import DialogueGCN
from models import DialogueGCN_DL
from models import Coach
from models import args
import helpers as utils


def main(args):
    utils.set_seed(args.seed)

    # load data
    log.debug("Loading data from '%s'." % args.data)
    data = utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = Dataset(data["train"], args.batch_size, multim=args.multim, audiovid=args.audiovid, audiotext=args.audiotext, textvid=args.textvid, exp3=args.exp3, decision_level = args.decision_level)
    devset = Dataset(data["dev"], args.batch_size, multim=args.multim, audiovid=args.audiovid, audiotext=args.audiotext, textvid=args.textvid, exp3=args.exp3, decision_level = args.decision_level)
    testset = Dataset(data["test"], args.batch_size, multim=args.multim, audiovid=args.audiovid, audiotext=args.audiotext, textvid=args.textvid, exp3=args.exp3, decision_level = args.decision_level)

    log.debug("Building model...")
    model_file = '/.../model.pt'
    model = DialogueGCN_DL(args).to(args.device) if args.decision_level else DialogueGCN(args).to(args.device)

    opt = Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)

    coach = Coach(trainset, devset, testset, model, opt, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)

    # Train.
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)

    return coach


def confusion_mat(y, y_pred, zero_diag: bool = False):
    matrix = confusion_matrix(y, y_pred)
    labels = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']

    # Plot
    if zero_diag == True:
        np.fill_diagonal(matrix, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    figure = sns.heatmap(matrix, annot=True, fmt="d", ax=ax)
    ax.set_xticklabels(labels, rotation=90, fontsize=15)
    ax.set_yticklabels(labels, rotation=0, fontsize=15)
    ax.set_xlabel("Predicted label", fontsize=15)
    ax.set_ylabel("True label", fontsize=15)
    plt.title("Confusion matrix", fontsize=20)
    plt.show()

    return matrix


def evaluate_model(coach, train=False, dev=False):
    model = coach.model
    dataset = coach.trainset if train else coach.devset if dev else coach.testset
    model.eval()
    with torch.no_grad():
        golds = []
        preds = []
        epoch_loss = 0
        for idx in tqdm(range(len(dataset)), desc="train" if train else "dev" if dev else 'test'):
            data = dataset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                data[k] = v.to(args.device)
            y_hat = model(data)
            nll = model.get_loss(data)
            epoch_loss += nll.item()
            preds.append(y_hat.detach().to("cpu"))

        golds = torch.cat(golds, dim=-1).numpy()
        preds = torch.cat(preds, dim=-1).numpy()

        # confusion matrix
        pl = confusion_mat(golds, preds)

        sns.set(style="white", font_scale=1.5)

        # loss plot
        plt.figure(figsize=(10, 7))
        plt.plot(coach.dev_hist['loss'], label='dev')
        plt.plot(coach.train_hist['loss'], label='train')
        plt.plot(coach.test_hist['loss'], label='test')
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title('Loss during training')
        plt.show()

        # f1 score plot
        plt.figure(figsize=(10, 7))
        plt.plot(coach.dev_hist['f1'], label='dev')
        plt.plot(coach.train_hist['f1'], label='train')
        plt.plot(coach.test_hist['f1'], label='test')
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("f1 score")
        plt.title('f1 score during training')
        plt.show()

        # accuracy plot
        plt.figure(figsize=(10, 7))
        plt.plot(coach.dev_hist['acc'], label='dev')
        plt.plot(coach.train_hist['acc'], label='train')
        plt.plot(coach.test_hist['acc'], label='test')
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title('Accuracy score during training')
        plt.legend()
        plt.show()

        # DEV RESULTS
        dev_f, dev_loss, dev_acc = coach.evaluate(test=False)
        log.info("DEVSET RESULTS: [F1_score: %f] [Loss: %f] [Acc: %f]" %
                 (dev_f, dev_loss, dev_acc))

        # TEST RESULTS
        test_f, test_loss, test_acc = coach.evaluate(test=True)
        log.info("DEVSET RESULTS: [F1_score: %f] [Loss: %f] [Acc: %f]" %
                 (test_f, test_loss, test_acc))

    return golds, preds


def get_preds(coach, train=False, dev=False):
    model = coach.model
    dataset = coach.trainset if train else coach.devset if dev else coach.testset
    model.eval()
    with torch.no_grad():
        golds = []
        preds = []
        epoch_loss = 0
        for idx in tqdm(range(len(dataset)), desc="train" if train else "dev" if dev else 'test'):
            data = dataset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                data[k] = v.to(args.device)
            y_hat = model(data)
            nll = model.get_loss(data)
            epoch_loss += nll.item()
            preds.append(y_hat.detach().to("cpu"))

        golds = torch.cat(golds, dim=-1).numpy()
        preds = torch.cat(preds, dim=-1).numpy()
    return golds, preds