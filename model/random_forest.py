import numpy as np
import torch
import torch.nn as nn

class Feature_Selection_Node(nn.Module):
    def __init__(self, number_of_trees, batch_size):
        super(Feature_Selection_Node, self).__init__()
        self.number_of_trees = number_of_trees
        self.batch = batch_size
        self.attention_mask = torch.nn.Parameter(data=torch.Tensor(number_of_trees, 28*28), requires_grad=True )
        self.attention_mask.data.uniform_(-1.0, 1.0)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        attention_tmp = torch.sigmoid(self.attention_mask)
        topk, idx = torch.topk(attention_tem, k = 200, dim=1)
        attention = torch.zeros(self.number_of_trees, 28*28)
        attention.scatter_(1, idx, topk)
        
        return_value = torch.zeros(self.batch, self.number_of_trees, 28*28)
        for mask_index in range(0, self.number_of_trees):
            return_value[:, mask_index, :] = x * attention[mask_index]
        return return_value, attention



class Decision_Node(nn.Module):
    def __init__(self, number_of_trees, max_num_leaf_nodes, classes, batch_size):
        super(Decision_Node, self).__init__()
        self.leaf = max_num_leaf_nodes
        self.tree = number_of_trees
        self.classes = classes
        self.batch = batch_size

        self.symbolic_path_weights = nn.Linear(28*28, max_num_leaf_nodes, bias=True)

        self.hardtanh = nn.Hardtanh()
        self.softmax = nn.Softmax(dim=-1)
        self.contribution = torch.nn.Parameter(data=torch.Tensor(number_of_trees, max_num_leaf_nodes, classes), requires_grad=True)
        self.contribution.data.uniform_(-1.0, -1.0)
    
    def forward(self, x):
        class_value = torch.randn(self.batch, self.tree, self.leaf, self.classes)
        symbolic_paths = self.hardtanh(self.symbolic_path_weights(x))

        for tree_index in range(0, self.tree):
            for decision_index in range(0, self.leaf):
                class_value[:, tree_index, decision_index, :] = torch.mm \
                    (symbolic_paths[:, tree_index, decision_index, :].view(-1, 1), self.contribution[tree_index, decision_index].view(1, -1))
        
        class_value = self.softmax(class_value)
        class_value = 1.0 - class_value * class_value
        class_value = class_value.sum(dim = -1)
        return symbolic_paths, class_value




if __name__ == '__main__':

    feature_sel = Feature_Selection_Node(10, 1)
    decision_sel = Decision_Node(10, 2, 2, 1)


        
