import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os.path

from collections import OrderedDict

from utils.early_stopping import EarlyStopping
from utils.logger import Logger


# set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


class BNView(nn.Module):
    def __init__(self, n_feat):
        super(BNView, self).__init__()
        self.n_feat = n_feat
        self.bn = nn.BatchNorm1d(n_feat)

    def forward(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], -1, self.n_feat)
        x = x.transpose(1, -1)
        x = self.bn(x)
        x = x.transpose(1, -1)
        x = x.reshape(x_shape)
        return x


def three_layer_mlp(input_dim, hidden_dim, output_dim, act="relu", dropout=0.0, bn=False):
    act_dict = {"relu": nn.ReLU, "tanh": nn.Tanh}
    layers = [
        ("fc1", nn.Linear(input_dim, hidden_dim)),
        ("act1", act_dict[act]()),
    ]
    assert dropout == 0.0 or not bn
    if dropout > 0:
        layers.append(("dropout1", nn.Dropout(dropout, inplace=True)))
    if bn:
        layers.append(("batchnorm1", BNView(hidden_dim)))
    layers.append(("fc2", nn.Linear(hidden_dim, output_dim)))
    return nn.Sequential(OrderedDict(layers))

class only_classify(nn.Module):
    def __init__(self):
        super(only_classify, self).__init__()
        self.bert_dim = 768
        self.embedding_dim = 32
        self.NUM_RULE = 1
        # if 'rule' and 'target' are generated from the same source
        # then, can use the same MLP
        self.rule_embedding = self.target_embedding = three_layer_mlp(
            input_dim=self.bert_dim,
            hidden_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
        )
        self.discovery_prob_embedding = three_layer_mlp(
            input_dim=self.embedding_dim * 2,
            hidden_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
        )

        self.judger = three_layer_mlp(
            input_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            output_dim=1,
        )
        self.probability = nn.Sigmoid()
        self.classification_loss = nn.BCELoss()
        self.judger_loss = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        # batch['rule'] (bz, 5, 1, 4, 768)
        #   there are 5 kinds of objects' rules
        #   every rule has 1 combination mode
        #   a mode is composed by 4 sub-rules
        #   a sub-rule has 768 features
        # batch['target'] (bz, 5, 1, 768)
        # batch['mask'] (bz, 5, 1, 4) alter
        #   if there is no mask available, that, all masks are 1.
        # batch['label'] (bz, 5)  alter
        assert {'rule', 'target'}.issubset(batch.keys()), "Please check [batch]."
        assert self.NUM_RULE == batch['rule'].shape[2], "The [NUM_RULE] is not matched."
        assert self.NUM_RULE == batch['target'].shape[2], "The [NUM_RULE] is not matched."
        batch_size = batch['rule'].shape[0]  # bz
        self.rule = batch['rule']  # (bz, 5, 1, 4, 768)
        self.target = batch['target']  # (bz, 5, 1, 768)
        if 'label' in batch.keys():
            self.label = batch['label']  # (bz, 5)
        if 'mask' in batch.keys():
            self.mask = batch['mask']  # (bz, 5, 1, 4)

        self.input = self.rule_embedding(self.rule)  # (bz, 5, 1, 4, 32)
        if 'mask' in batch.keys():
            self.output = self.input * self.mask.unsqueeze(-1).expand_as(self.input)
            self.rule_output = torch.mean(self.output, dim=-2)
        else:
            self.rule_output = torch.mean(self.input, dim=-2)  # (bz, 5, 1, 32)

        # self.target_embedding=self.rule_embedding MLP(768, 32, 32)
        self.target_embed = self.target_embedding(self.target)  # (bz, 5, 1, 32)

        self.target_output = self.discovery_prob_embedding(
            torch.cat((self.rule_output, self.target_embed), dim=-1)
        )  # (bz, 5, 1, 32)

        vote = self.judger(self.target_output).squeeze(-1)  # (bz, 5, 1)
        self.s = torch.mean(vote, dim=-1)  # (bz, 5)

        self.p = self.probability(self.s)

        output = {}
        output['s'] = self.s
        output['p'] = self.p

        output['L1'] = self.compute_l1_regularization()
        if 'label' in batch.keys():
            output['L_cls'] = self.classification_loss(self.p, self.label)
            output['loss'] = output['L_cls'] + output['L1']
        else:
            output['L_cls'] = torch.tensor(-1, dtype=torch.float)
            output['loss'] = output['L1']

        return output

    def compute_l1_regularization(self, lambda_l1=1e-5):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.norm(param, 1)
        return lambda_l1 * l1_loss

    @staticmethod
    def model_train(net, dataloader, max_epoch=200, path='./pth/logical_reason.pt'):
        lr_list = []
        logger = Logger(tensorboard=True, matplotlib=True, log_dir=os.path.dirname(path))

        early_stopping = EarlyStopping(20, verbose=False, delta=1e-2)
        net.train()

        optimizer = optim.AdamW(
            net.parameters(),
            lr=5e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epoch
        )

        for epoch in range(max_epoch):
            for batch in dataloader:
                output = net(batch)
                loss = output['loss']
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                optimizer.step()
            scheduler.step()
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

            # for name, param in net.named_parameters():
            #     if param.grad is None:
            #         print(f"No gradient for: {name}")

            print(f"Epoch is {epoch+1:>3}/{max_epoch}, total loss is {loss.item():.6f}, "
                  f"l_cls is {output['L_cls'].item():.6f}, L1 is {output['L1'].item():.6f}")

            logger_dict = {"epoch": epoch, "loss": loss, "L_cls": output['L_cls'], "lr": scheduler.get_lr()}
            logger.update_scalers(logger_dict)

            if output['L_cls'].item() < 1e-3:
                # net.check_acc_by_judge()
                early_stopping(loss, net, path=path)
                if early_stopping.early_stop:
                    print("Early stopping.")
                    break

    @staticmethod
    def model_eval(net, batch):
        net.eval()
        output = net(batch)
        # print(f"{output['s']=}")
        # print(f"{output['p']=}")
        max_values, max_indices = torch.max(output['p'], dim=1)
        valid_mask = max_values > 0.8
        result = [max_indices[i].item() if valid_mask[i] else None for i in range(output['p'].shape[0])]
        return result
