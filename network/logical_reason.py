import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MHAtt(nn.Module):
    def __init__(self, n_heads=4, dim=32, dropout=0.0):
        super(MHAtt, self).__init__()
        self.n_head, self.dim = n_heads, dim
        self.query, self.key, self.value = (
            nn.Linear(dim, n_heads * dim) for _ in range(3)
        )
        self.linear = nn.Linear(n_heads * dim, dim)
        self.att_norm = nn.LayerNorm(dim)
        self.ffn = three_layer_mlp(dim, 4 * dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (bz,18,10,k,32)
        prev_shape = x.shape[:-2]  # (bz,18,10)
        query = self.query(x).reshape(*prev_shape, -1, self.n_head, self.dim) # (bz,18,10,k,4,32)
        key = self.key(x).reshape(*prev_shape, -1, self.n_head, self.dim) # (bz,18,10,k,4,32)
        value = self.value(x).reshape(*prev_shape, -1, self.n_head, self.dim) # (bz,18,10,k,4,32)

        query, key, value = (
            query.transpose(-2, -3),
            key.transpose(-2, -3),
            value.transpose(-2, -3),
        ) # (bz,18,10,4,k,32)

        att = F.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / int(np.sqrt(self.dim)), dim=-1
        ) # (bz,18,10,4,k,32)*(bz,18,10,4,32,k) = (bz,18,10,4,k,k)
        linear = self.linear(
            torch.matmul(att, value)
                .transpose(-2, -3)
                .reshape(*prev_shape, -1, self.n_head * self.dim)
        ) # (bz,18,10,4,k,k)*(bz,18,10,4,k,32) = (bz,18,10,4,k,32)->(bz,18,10,k,4,32)->(bz,18,10,k,4*64)->(bz,18,10,k,32)
        att_norm = self.att_norm(x + linear) # (bz,18,10,k,32)

        # self.ffn(att_norm) three_layer_mlp(32, 4*32, 32)
        ffn = self.ffn_norm(att_norm + self.ffn(att_norm))

        return ffn  # (bz,18,10,k,32)


class logical_reason(nn.Module):
    def __init__(self, l_logic_fac: float=None):
        super(logical_reason, self).__init__()
        self.bert_dim = 768
        self.embedding_dim = 32
        self.NUM_RULE = 1
        self.l_cls_fac = 1.0  # 1.0, 0.2
        self.l_logic_fac = 0.2 if l_logic_fac is None else l_logic_fac
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
        self.not_layers = three_layer_mlp(
            input_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
        )
        self.or_attn = MHAtt(dim=self.embedding_dim)
        self.mh_att = MHAtt(dim=self.embedding_dim)
        self.vote_final = three_layer_mlp(
            input_dim=self.NUM_RULE * self.embedding_dim,
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

    def not_module(self, input):
        # input (bz, 5, 1, 4, 32)
        return self.not_layers(input)  # (bz, 5, 1, 4, 32)

    def or_module(self, input):
        if input.shape[-1] == self.embedding_dim:
            # input (bz, 5, 1, 4, 32)
            return self.or_attn(input).mean(-2)  # (bz, 5, 1, 32)
        else:
            # input (bz, 5, 1, 32*4)
            return self.or_attn(
                input.reshape(*input.shape[:-1], -1, self.embedding_dim)
            ).mean(-2)

    def logic_loss_x_or_x(self, x):
        # x or x == x
        reg = self.judger_loss(
            self.judger(self.or_module(torch.cat([x, x], dim=-1))),
            self.judger(x).sigmoid()
        )
        return reg

    def logic_loss_x_or_not_x(self, x):
        # x or (not x) == T
        # T(embed_dim) -> T(x.shape),
        reg = self.judger_loss(self.judger(self.or_module(torch.cat([x, self.not_module(x)], dim=-1))).squeeze(-1),
                               x.new_ones(x.shape[:-1]))  # x.new_ones(x.shape[:-1])  # x.new_zeros(x.shape[:-1])
        return reg

    def logic_loss_x_or_T(self, x):
        # x or T == T
        reg = self.judger_loss(
            self.judger(self.or_module(torch.cat([x, x.new_ones(x.shape)], dim=-1))).squeeze(-1),
            x.new_ones(x.shape[:-1])
        )
        return reg

    def logic_loss_x_or_F(self, x):
        # x or F == x
        reg = self.judger_loss(
            self.judger(self.or_module(torch.cat([x, x.new_zeros(x.shape)], dim=-1))),
            self.judger(x).sigmoid()
        )
        return reg

    def logic_loss(self):
        T_in = self.pos_input.new_ones(self.pos_input.shape)
        F_in = self.pos_input.new_zeros(self.pos_input.shape)

        # not x == 1 - x
        reg_1 = self.judger_loss(self.judger(self.not_output), 1 - self.judger(self.pos_input).sigmoid())
        # not+not+x == x
        not_not_output = self.not_module(self.not_output)
        reg_2 = self.judger_loss(self.judger(not_not_output), self.judger(self.pos_input).sigmoid())

        # x or T == T
        # not_output(bz,5,1,4,32), or_target_output(bz,5,1,32), target_embed(bz,5,1,32)
        or_reg_targets = [self.not_output, self.or_target_output, self.target_embed]

        # print([reg_target.shape for reg_target in or_reg_targets])
        # reg_3 = sum(self.logic_loss_x_or_T(reg_target) for reg_target in or_reg_targets[:-2])
        # reg_4 = sum(self.logic_loss_x_or_F(reg_target) for reg_target in or_reg_targets[:-2])

        reg_5 = sum(self.logic_loss_x_or_x(reg_target) for reg_target in or_reg_targets)
        reg_6 = sum(self.logic_loss_x_or_not_x(reg_target) for reg_target in or_reg_targets)

        # F or F == F
        reg_7 = self.judger_loss(self.judger(self.or_module(torch.cat([F_in, F_in], dim=-1))).squeeze(-1),
                               F_in[..., 0])
        # not T == F
        reg_8 = self.judger_loss(self.judger(self.not_module(T_in)).squeeze(-1), F_in[..., 0])
        # T or F == T
        reg_9 = self.judger_loss(self.judger(self.or_module(torch.cat([T_in, F_in], dim=-1))).squeeze(-1),
                               T_in[..., 0])

        # dic = {1: reg_1, 2: reg_2, 5: reg_5, 6: reg_6}
        dic = {1: reg_1, 2: reg_2, 5: reg_5, 6: reg_6, 7: reg_7, 8: reg_8, 9: reg_9}
        reg = sum(dic.values())
        # print(dic)

        return reg

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

        self.pos_input = self.rule_embedding(self.rule)  # (bz, 5, 1, 4, 32)
        self.not_output = self.not_module(self.pos_input)  # (bz, 5, 1, 4, 32)
        # self.not_output = self.pos_input  # and method
        if 'mask' in batch.keys():
            # self.not_output = self.not_output * self.mask.unsqueeze(-1).expand_as(self.not_output)
            self.not_output = self.not_module(self.pos_input * self.mask.unsqueeze(-1).expand_as(self.pos_input))  # new mask
            self.or_rule_output = self.or_module(self.not_output)
        else:
            self.or_rule_output = self.or_module(self.not_output)  # (bz, 5, 1, 32)

        self.or_rule_output += torch.mean(self.not_output, dim=-2)  # new res

        # self.target_embedding=self.rule_embedding MLP(768, 32, 32)
        self.target_embed = self.target_embedding(self.target)  # (bz, 5, 1, 32)

        self.or_target_output = self.or_module(
            torch.cat([self.target_embed, self.or_rule_output], dim=-1)
        )  # (bz, 5, 1, 32)

        vote = self.judger(self.or_target_output).squeeze(-1)  # (bz, 5, 1)
        self.s = torch.mean(vote, dim=-1)  # (bz, 5)

        self.p = self.probability(self.s)

        output = {}
        output['s'] = self.s
        output['p'] = self.p

        output['L_logic'] = self.logic_loss()
        output['L1'] = self.compute_l1_regularization()
        if 'label' in batch.keys():
            output['L_cls'] = self.classification_loss(self.p, self.label)
            output['loss'] = self.l_cls_fac * output['L_cls'] + self.l_logic_fac * output['L_logic'] + output['L1']
        else:
            output['L_cls'] = torch.tensor(-1, dtype=torch.float)
            output['loss'] = output['L_logic']

        return output

    def check_acc_by_judge(self):
        def _c(rule, target, threshold=0.5):
            return (torch.abs(rule - target) < threshold).sum().item()

        max_epoch = 50
        bz = 64
        counts = [torch.tensor([0, 0]) for i in range(14)]
        count_1, count_2, count_3, count_4, count_5, count_6, count_7, count_8, count_9, count_10, count_11, count_12, count_13, count_14 = counts
        with torch.no_grad():
            for epoch in range(max_epoch):
                x = self.rule_embedding(torch.randn((bz, 1, 1, self.bert_dim)))
                not_x = self.not_module(x)
                not_not_x = self.not_module(not_x)
                T = torch.ones_like(x)
                F = torch.zeros_like(x)

                # x == 1 - not(x)
                count_1[0] += _c(self.judger(x).sigmoid(), 1 - self.judger(not_x).sigmoid())
                count_1[1] += _c(self.judger(x).sigmoid(), self.judger(not_x).sigmoid())

                # x == not(not(x))
                count_2[0] += _c(self.judger(x).sigmoid(), self.judger(not_not_x).sigmoid())
                count_2[1] += _c(self.judger(x).sigmoid(), 1 - self.judger(not_not_x).sigmoid())

                # x or x == x
                count_3[0] += _c(self.judger(self.or_module(torch.cat([x, x], dim=-1))).sigmoid(), self.judger(x).sigmoid())
                count_3[1] += _c(self.judger(self.or_module(torch.cat([x, x], dim=-1))).sigmoid(), 1 - self.judger(x).sigmoid())

                # x or not(x) == T
                count_4[0] += _c(self.judger(self.or_module(torch.cat([x, not_x], dim=-1))).sigmoid(), T[..., :1])
                count_4[1] += _c(self.judger(self.or_module(torch.cat([x, not_x], dim=-1))).sigmoid(), 1 - T[..., :1])

                # T or T == T
                count_5[0] += _c(self.judger(self.or_module(torch.cat([T, T], dim=-1))).sigmoid(), T[..., :1])
                count_5[1] += _c(self.judger(self.or_module(torch.cat([T, T], dim=-1))).sigmoid(), 1 - T[..., :1])

                # T or F == T
                count_6[0] += _c(self.judger(self.or_module(torch.cat([T, F], dim=-1))).sigmoid(), T[..., :1])
                count_6[1] += _c(self.judger(self.or_module(torch.cat([T, F], dim=-1))).sigmoid(), 1 - T[..., :1])

                # F or T == T
                count_7[0] += _c(self.judger(self.or_module(torch.cat([F, T], dim=-1))).sigmoid(), T[..., :1])
                count_7[1] += _c(self.judger(self.or_module(torch.cat([F, T], dim=-1))).sigmoid(), 1 - T[..., :1])

                # F or F == F
                count_8[0] += _c(self.judger(self.or_module(torch.cat([F, F], dim=-1))).sigmoid(), F[..., :1])
                count_8[1] += _c(self.judger(self.or_module(torch.cat([F, F], dim=-1))).sigmoid(), 1 - F[..., :1])

                # T or T or T == T
                count_9[0] += _c(self.judger(self.or_module(torch.cat([T, T, T], dim=-1))).sigmoid(), T[..., :1])
                count_9[1] += _c(self.judger(self.or_module(torch.cat([T, T, T], dim=-1))).sigmoid(), 1 - T[..., :1])

                # T or F or T == T
                count_10[0] += _c(self.judger(self.or_module(torch.cat([T, F, T], dim=-1))).sigmoid(), T[..., :1])
                count_10[1] += _c(self.judger(self.or_module(torch.cat([T, F, T], dim=-1))).sigmoid(), 1 - T[..., :1])

                # T or F or F == T
                count_11[0] += _c(self.judger(self.or_module(torch.cat([T, F, F], dim=-1))).sigmoid(), T[..., :1])
                count_11[1] += _c(self.judger(self.or_module(torch.cat([T, F, F], dim=-1))).sigmoid(), 1 - T[..., :1])

                # F or F or F == F
                count_12[0] += _c(self.judger(self.or_module(torch.cat([F, F, F], dim=-1))).sigmoid(), F[..., :1])
                count_12[1] += _c(self.judger(self.or_module(torch.cat([F, F, F], dim=-1))).sigmoid(), 1 - F[..., :1])

                # x or F == x
                count_13[0] += _c(self.judger(self.or_module(torch.cat([x, F], dim=-1))).sigmoid(), self.judger(x).sigmoid())
                count_13[1] += _c(self.judger(self.or_module(torch.cat([x, F], dim=-1))).sigmoid(), 1 - self.judger(x).sigmoid())

                # x or T == T
                count_14[0] += _c(self.judger(self.or_module(torch.cat([x, T], dim=-1))).sigmoid(), T[..., :1])
                count_14[1] += _c(self.judger(self.or_module(torch.cat([x, T], dim=-1))).sigmoid(), 1 - T[..., :1])

        length = max_epoch * bz
        for i in range(len(counts)):
            x = locals()[f'count_{i + 1}']
            print(f'ACC_{i + 1:<2}: {x[0] / length * 100:<7.3f}%, {x[1] / length * 100:<7.3f}%')

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
                # loss = output['L_cls'] + output['L1']
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
                  f"l_cls is {output['L_cls'].item():.6f}, l_logic is {output['L_logic'].item():.6f}, "
                  f"L1 is {output['L1'].item():.6f}")


            logger_dict = {"epoch": epoch, "loss": loss, "L_cls": output['L_cls'], "L_logic": output['L_logic'],
                           "lr": scheduler.get_lr()}
            logger.update_scalers(logger_dict)

            if output['L_cls'].item() < 1e-3:
                # net.check_acc_by_judge()
                early_stopping(loss, net, path=path)
                if early_stopping.early_stop:
                    print("Early stopping.")
                    break

        net.eval()
        net.check_acc_by_judge()

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
