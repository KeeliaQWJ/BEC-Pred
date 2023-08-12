import torch
import torch.nn as nn

import class_predictor.graph.mol_features as mol_features

import pdb


class RxnPredictor(nn.Module):
    def __init__(self, args, n_classes=1):
        super(RxnPredictor, self).__init__()
        self.args = args

        self.src_graph_conv = GraphConv(args)
        if args.share_embed:
            self.tgt_graph_conv = self.src_graph_conv
        else:
            self.tgt_graph_conv = GraphConv(args)

        hidden_size = args.hidden_size
        self.r_h = nn.Linear(hidden_size, hidden_size)
        self.r_o = nn.Linear(hidden_size, n_classes)

    def aggregate_atom_h(self, atom_h, scope):
        mol_h = []
        for (st, le) in scope:
            cur_atom_h = atom_h.narrow(0, st, le)
            mol_h.append(cur_atom_h.sum(dim=0))
        mol_h = torch.stack(mol_h, dim=0)
        return mol_h

    def get_graph_embeds(self, graphs, conv_model):
        graph_inputs, scope = graphs.get_graph_inputs()

        atom_h = conv_model(graph_inputs)
        mol_h = self.aggregate_atom_h(atom_h, scope)
        return mol_h

    def forward(self, src_graphs, tgt_graphs, args):
        src_mol_h = self.get_graph_embeds(src_graphs, self.src_graph_conv)
        tgt_mol_h = self.get_graph_embeds(tgt_graphs, self.tgt_graph_conv)

        rxn_h = tgt_mol_h - src_mol_h
        rxn_h = nn.ReLU()(self.r_h(rxn_h))
        rxn_o = self.r_o(rxn_h)
        return rxn_o


class GraphConv(nn.Module):
    def __init__(self, args):
        """Creates graph conv layers for molecular graphs."""
        super(GraphConv, self).__init__()
        self.args = args
        hidden_size = args.hidden_size

        self.n_atom_feats = mol_features.N_ATOM_FEATS
        self.n_bond_feats = mol_features.N_BOND_FEATS

        # Weights for the message passing network
        self.W_message_i = nn.Linear(self.n_atom_feats + self.n_bond_feats,
                                     hidden_size, bias=False,)
        self.W_message_h = nn.Linear(hidden_size, hidden_size, bias=False,)
        self.W_message_o = nn.Linear(self.n_atom_feats + hidden_size,
                                     hidden_size)

        self.dropout = nn.Dropout(args.dropout)

    def index_select_nei(self, input, dim, index):
        # Reshape index because index_select expects a 1-D tensor. Reshape the
        # output afterwards.
        target = torch.index_select(
            input=input,
            dim=0,
            index=index.view(-1)
        )
        return target.view(index.size() + input.size()[1:])

    def forward(self, graph_inputs):
        fatoms, fbonds, agraph, bgraph = graph_inputs

        # nei_input_h is size [# bonds, hidden_size]
        nei_input_h = self.W_message_i(fbonds)
        # message_h is size [# bonds, hidden_size]
        message_h = nn.ReLU()(nei_input_h)

        for i in range(self.args.depth - 1):
            # nei_message_h is [# bonds, # max neighbors, hidden_size]
            nei_message_h = self.index_select_nei(
                input=message_h,
                dim=0,
                index=bgraph)

            # Sum over the nieghbors, now [# bonds, hidden_size]
            nei_message_h = nei_message_h.sum(dim=1)
            nei_message_h = self.W_message_h(nei_message_h)  # Shared weights

            message_h = nn.ReLU()(nei_input_h + nei_message_h)

        # Collect the neighbor messages for atom aggregation
        nei_message_h = self.index_select_nei(
            input=message_h,
            dim=0,
            index=agraph,
        )
        # Aggregate the messages
        nei_message_h = nei_message_h.sum(dim=1)
        atom_input = torch.cat([fatoms, nei_message_h], dim=1)
        atom_input = self.dropout(atom_input)

        atom_h = nn.ReLU()(self.W_message_o(atom_input))
        return atom_h


#这段代码定义了一个用于预测化学反应的神经网络模型。该模型基于图卷积神经网络（Graph Convolutional Neural Network，GCN）
# ，用于处理分子图结构的数据。该模型包含两个主要的类：RxnPredictor和GraphConv。
# RxnPredictor是用于预测化学反应的主体神经网络类，而GraphConv是用于构建分子图结构的图卷积层。 
 
# 1. 首先，导入所需的库： 
#    - torch：PyTorch库，用于实现神经网络模型。 
#    - torch.nn：PyTorch库中的神经网络模块。 
#    - mol_features：用于处理分子图特征的模块。 
#    - pdb：Python调试器。 
# 2. 定义RxnPredictor类，它继承自nn.Module。这是主体神经网络模型，用于预测化学反应。  
# 3. 定义RxnPredictor类的__init__()方法，初始化模型参数和图卷积层（src_graph_conv和tgt_graph_conv）。  
# 4. 定义aggregate_atom_h()方法，用于汇总原子表示（atom_h）以获得分子表示（mol_h）。这个方法将用于处理图卷积层的输出。 
# 5. 定义get_graph_embeds()方法，用于获取分子图的嵌入表示。这个方法将图卷积模型（conv_model）应用于输入的图结构数据（graphs）。  
# 6. 定义RxnPredictor类的forward()方法，根据输入的源分子图（src_graphs）和目标分子图（tgt_graphs）计算化学反应。 
# 7. 定义GraphConv类，它继承自nn.Module。这是用于构建图卷积层的类。  
# 8. 定义GraphConv类的__init__()方法，初始化图卷积层的参数和权重矩阵。  
# 9. 定义index_select_nei()方法，用于从输入矩阵中选择邻居节点的表示。 
# 10. 定义GraphConv类的forward()方法，实现图卷积神经网络层的计算过程。  
# 要运行这段代码，首先需要确保已经安装了PyTorch库。接下来，你需要创建一个脚本文件（例如，main.py），并将这段代码粘贴到文件中。
# 然后，在同一目录下创建一个包含mol_features模块的文件（例如，mol_features.py）。
# 接下来，在main.py文件中实例化RxnPredictor类并使用适当的输入数据调用其forward方法，如下所示：
# import torch
# from main import RxnPredictor, GraphConv
# from data import src_graphs, tgt_graphs, args

# # 实例化RxnPredictor模型
# model = RxnPredictor(args)

# # 转换输入数据为PyTorch张
# src_graphs_tensor = torch.tensor(src_graphs)
# tgt_graphs_tensor = torch.tensor(tgt_graphs)

# # 运行模型
# output = model.forward(src_graphs_tensor, tgt_graphs_tensor, args)
# 在这个例子中， src_graphs 和 tgt_graphs 分别表示源分子图和目标分子图的数据，而 args 是一个包含模型参数的对象。注意，你需要根据实际情况修改输入数据和参数对象