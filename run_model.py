import os
import shutil
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from mlmodel import Wavelet_model as create_model
from mlmodel import MyDataSet, train_one_epoch, evaluate

class para_init():
    def __init__(self,
                embed_dim=20, 
                power_exponent=1,
                poly_degree=1, 
                nnlist_length=100,
                feature_update_depth=1, 

                epochs=20,
                train_batch_size=8,
                test_batch_size=1,
                lr=0.001,
                lrf=0.001,
                device='cuda:0',

                dataset_folder="~/dataset_folder"):
        self.embed_dim = embed_dim
        self.power_exponent = power_exponent
        self.poly_degree = poly_degree
        self.nnlist_length = nnlist_length
        self.feature_update_depth = feature_update_depth

        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.lrf = lrf
        self.device = device

        self.dataset_folder = dataset_folder


def model_run(args, train_x, train_y, test_x, test_y):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = MyDataSet(train_x, train_y)
    testing_dataset = MyDataSet(test_x, test_y)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 16])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.train_batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=testing_dataset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=testing_dataset.collate_fn)
    
    # 模型参数设置
    model = create_model(embed_dim=args.embed_dim,
                        power_exponent=args.power_exponent,
                        poly_degree=args.poly_degree,
                        nnlist_length=args.nnlist_length,
                        feature_update_depth=args.feature_update_depth
                         ).to(device)
    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(pg, lr=args.lr,)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

    for epoch in range(args.epochs):

        train_loss = train_one_epoch(model=model,
                                            optimizer=optimizer,
                                            data_loader=train_loader,
                                            device=device,
                                            epoch=epoch)
        
        nlp_pre, nlp_ture = evaluate(model=model,
                                data_loader=test_loader,
                                device=device,
                                epoch=epoch)

        scheduler.step()



if __name__ == '__main__':
    
    root_dir = os.getcwd()
    model_para_init = para_init()
    
    





