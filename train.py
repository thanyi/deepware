import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset.dataset import DeepfakeDataset
import dataset.dataset_conf as config
from my_eval import prepare_model
from utils.utils import evaluate


def f3net_training():
    device = torch.device('cuda')

    train_data = DeepfakeDataset(normal_root=config.normal_root, malicious_root=config.malicious_root, mode='train', resize=380,
                                 csv_root=config.csv_root)
    train_data_size = len(train_data)
    print('train_data_size:', train_data_size)

    train_loader = DataLoader(train_data, 16, shuffle=True)

    # train

    model = prepare_model()
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0002, betas=(0.9, 0.999))


    times = 0
    for epoch in range(1, 10):
        print("第{}个epoch".format(epoch))
        train_step = 0

        for i, (X, y) in enumerate(train_loader):
            model.train()

            X = X.to(device)
            y = y.to(device)

            output = model(X)
            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step += 1
            if train_step %10 ==0:
                print("第{}个batch训练完成".format(train_step))
                print("Loss: {}".format(loss.item()))


        print("epoch训练次数：{}, Loss: {}".format(model.total_steps, model.loss.item()))

        if epoch % 1 == 0:
            times += 1
            torch.save(model.model.state_dict(),
                       "/content/drive/MyDrive/models/F3/test_6(git_version)/model{}.pth".format(times))


            model.model.eval()

            r_acc, auc = evaluate(model, config.normal_root, config.malicious_root, config.csv_root, "valid")
            print("本次epoch的acc为：" + str(r_acc))
            print("本次epoch的auc为：" + str(auc))
            model.model.train()





if __name__ == '__main__':
    f3net_training()
