from pymodule import *
from network import *

def get_parser():
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--train_initial_model', action='store_true', default=True)     
    parser.add_argument('--optimize_num', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--train_new_model', action='store_true', default=False)
    parser.add_argument('--max_f1_val', type=float, default=0.1)  

    # network para
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)  
    parser.add_argument('--lr', type=float, default=0.08) 
    parser.add_argument('--weight_decay', type=float, default=1e-5)  
    parser.add_argument('--hard_label', action='store_true', default=True)

    # hyper para
    parser.add_argument('--T_confidence', type=float, default=0.85)
    parser.add_argument('--T_influence', type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--max_nodes_num', type=int, default=30)    

    # data set 
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--imbalance_ratio', type=float, default=0.1)
    parser.add_argument('--minor_class_num', type=int, default=3) 
    parser.add_argument('--val_num', type=int, default=500)
    parser.add_argument('--test_num', type=int, default=1000)

    return parser



def load_data(args,path="../data"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """

    print("\nUpload {} dataset.".format(args.dataset))
    if args.dataset == 'ogbn':
        f=open(path+'/ogbn/graph.pkl', 'rb')
        graph=pickle.load(f)
        f=open(path+'/ogbn/features.pkl', 'rb')
        features=pickle.load(f)
        f=open(path+'/ogbn/labels.pkl', 'rb')
        labels=pickle.load(f)

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj_original=adj
        adj = normalize_adj(adj + sp.eye(adj.shape[0])) 
        adj = torch.FloatTensor(np.array(adj.todense()))

        features = torch.FloatTensor(features)
        
        idx_train, idx_val, idx_test, idx_unlabeled = dataset_div_ogbn(args, args.val_num, args.test_num, labels, args.imbalance_ratio, args.minor_class_num)
        
        labels_onehot=one_hot(labels, int(max(labels)+1))
        labels = torch.IntTensor(labels)

    elif args.dataset == 'computers':

        target_dataset = Amazon(path, name='Computers')

        target_data = target_dataset[0]
        target_data.num_classes = np.max(target_data.y.numpy())+1

        mask_list = [i for i in range(target_data.num_nodes)]
        random.seed(0)
        random.shuffle(mask_list)

        train_mask_list, valid_mask_list, test_mask_list, target_data.train_node, unlabeled_mask_list = get_split(
            args, mask_list, target_data.y.numpy(), nclass=target_data.num_classes)

        idx_train = torch.tensor(train_mask_list)
        idx_val = torch.tensor(valid_mask_list)
        idx_test = torch.tensor(test_mask_list)
        idx_unlabeled = torch.tensor(unlabeled_mask_list)

        features = target_data.x
        G = to_networkx(target_data, to_undirected=True)
        adj = nx.adjacency_matrix(G)
        adj = adj + sp.eye(adj.shape[0])
        adj_original = adj
        adj = normalize_adj(adj)
        adj = torch.FloatTensor(np.array(adj.todense()))
        features = normalize(features)
        features = torch.FloatTensor(features)
        labels = target_data.y
        labels_onehot = one_hot(labels, int(max(labels)+1))

    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []

        for i in range(len(names)):
            with open("{}/ind.{}.{}".format(path, args.dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pickle.load(f, encoding='latin1'))
                else:
                    objects.append(pickle.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        
        test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, args.dataset))
        test_idx_range = np.sort(test_idx_reorder)
        
        if args.dataset == 'citeseer':
            #Citeseer dataset contains some isolated nodes in the graph
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended

            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))        
        
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj_original=adj
        adj = normalize_adj(adj + sp.eye(adj.shape[0])) 
        adj = torch.FloatTensor(np.array(adj.todense()))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        if args.dataset == 'citeseer':
            save_label = np.where(labels)[1]
        labels = torch.LongTensor(np.where(labels)[1])

        idx_test = list(test_idx_range.tolist())
        
        def missing_elements(L):
            start, end = L[0], L[-1]
            return sorted(set(range(start, end+1)).difference(L))

        if args.dataset == 'citeseer':
            L = np.sort(idx_test)
            missing = missing_elements(L)

            for element in missing:
                save_label = np.insert(save_label, element, 0)

            labels = torch.LongTensor(save_label)

        labels_onehot=one_hot(labels, ally.shape[1])

        idx_train, idx_val, idx_test, idx_unlabeled = dataset_div(args, y, args.val_num, args.test_num, labels)
        idx_train=under_sampling(idx_train, labels, args.imbalance_ratio, args.minor_class_num)
    
    class_num=labels.max().item()+1
    class_counter=[]
    for cla in range(class_num):
        class_counter.append(labels[idx_train].tolist().count(cla))
    print("\nTraining Node numbers:", class_counter,'Sum:',sum(class_counter) )

    class_counter=[]
    for cla in range(class_num):
        class_counter.append(labels[idx_val].tolist().count(cla))
    print("Validation Node numbers:", class_counter,'Sum:',sum(class_counter) )

    class_counter=[]
    for cla in range(class_num):
        class_counter.append(labels[idx_test].tolist().count(cla))
    print("Testing Node numbers:", class_counter,'Sum:',sum(class_counter) )

    class_counter=[]
    for cla in range(class_num):
        class_counter.append(labels[idx_unlabeled].tolist().count(cla))
    print("Unlabeled Node numbers:", class_counter,'Sum:',sum(class_counter) )

    return adj, features, labels, labels_onehot, idx_train, idx_val, idx_test, idx_unlabeled, adj_original


def get_split(args, all_idx, all_label, nclass=10):
    train_each = 20  
    valid_each = 50 
    test_each = 100 
    minor_num = args.imbalance_ratio*train_each
    all_num_node = (train_each+minor_num)*(nclass/2)

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if iter_label < (nclass/2):
            if train_list[iter_label] < train_each:
                train_list[iter_label] += 1
                train_node[iter_label].append(iter1)
                train_idx.append(iter1)
        else:
            if train_list[iter_label] < minor_num:
                train_list[iter_label] += 1
                train_node[iter_label].append(iter1)
                train_idx.append(iter1)
        if sum(train_list) == all_num_node:
            break
    assert sum(train_list) == all_num_node
    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < valid_each:
            valid_list[iter_label] += 1
            valid_idx.append(iter2)
        if sum(valid_list) == valid_each*nclass:
            break

    assert sum(valid_list) == valid_each*nclass
    after_val_idx = list(set(after_train_idx)-set(valid_idx))

    test_list = [0 for _ in range(nclass)]
    test_idx = []
    for iter3 in after_val_idx:
        iter_label = all_label[iter3]
        if test_list[iter_label] < test_each:
            test_list[iter_label] += 1
            test_idx.append(iter3)
        if sum(test_list) == test_each*nclass:
            break

    assert sum(test_list) == test_each*nclass
    unlabeled_idx = list(set(after_val_idx)-set(test_idx))
    return train_idx, valid_idx, test_idx, train_node, unlabeled_idx

def dataset_div_ogbn(args, val_num, test_num, labels,imbalance_ratio, minor_class_num):

    class_num=labels.max().item() +1
    classes_0={}

    for cla in range(class_num):
        classes_0[cla]=[]

    for i in range(labels.size):
        classes_0[labels.tolist()[i]].append(i)
        
    train_class_num=int(20)
    val_class_num=int(val_num/class_num)
    test_class_num=int(test_num/class_num)
    random.seed(1024)
    idx_train=[]
    idx_val=[]
    idx_test=[]
    idx_unlabeled=[] 
    for cla in range(class_num):
        random.shuffle(classes_0[cla])
        idx_train.extend(classes_0[cla][0:train_class_num])
        idx_val.extend(classes_0[cla][train_class_num:train_class_num+val_class_num])
        idx_test.extend(classes_0[cla][train_class_num+val_class_num:train_class_num+val_class_num+test_class_num])       
        idx_unlabeled.extend(classes_0[cla][train_class_num+val_class_num+test_class_num:])
    idx_train, idx_val, idx_test, idx_unlabeled= list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test, idx_unlabeled]))

  
    minor_num=20*imbalance_ratio
    buf=[]
    for i in range(minor_class_num):
        num=0
        cla=i+(labels.max().item()+1-minor_class_num)
        for j in range(idx_train.size()[0]):
            if labels[idx_train[j]]==cla:
                num+=1
            if labels[idx_train[j]]==cla and num>minor_num:
                buf.append(j)
    idx_train_new=torch.from_numpy(np.delete(np.array(idx_train), buf,0))

    class_counter=[]
    for cla in range(class_num):
        class_counter.append(labels[idx_train_new].tolist().count(cla))
    print("Node numbers:", class_counter)

    return idx_train_new, idx_val, idx_test, idx_unlabeled 


def dataset_div(args, y, val_num, test_num, labels):
    idx_train = list(range(len(y)))
    class_num=labels.max().item() +1

    idx_others=[]
    for i in range(labels.size()[0]):
        if i not in idx_train:
            idx_others.append(i)
    random.seed(1024)
    random.shuffle(idx_others)   
    idx_val=idx_others[0:val_num]
    idx_test=idx_others[val_num:val_num+test_num]
    idx_unlabeled=idx_others[val_num+test_num:]
    idx_train, idx_val, idx_test, idx_unlabeled= list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test, idx_unlabeled]))

    return idx_train, idx_val, idx_test, idx_unlabeled 


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def label_processing(y, ty, ally,minor_class):

    y_pro=np.hstack((y[:,minor_class].reshape(-1,1),(np.ones((y.shape[0],))-y[:,minor_class]).reshape(-1,1)))
    ty_pro=np.hstack((ty[:,minor_class].reshape(-1,1),(np.ones((ty.shape[0],))-ty[:,minor_class]).reshape(-1,1)))
    ally_pro=np.hstack((ally[:,minor_class].reshape(-1,1),(np.ones((ally.shape[0],))-ally[:,minor_class]).reshape(-1,1)))
    return y_pro, ty_pro, ally_pro


def under_sampling(idx_train, labels, imbalance_ratio, minor_class_num):
    minor_num=20*imbalance_ratio
    buf=[]
    for i in range(minor_class_num):
        num=0
        cla=i+(labels.max().item()+1-minor_class_num)
        for j in range(idx_train.size()[0]):
            if labels[idx_train[j]]==cla:
                num+=1
            if labels[idx_train[j]]==cla and num>minor_num:
                buf.append(j)
    idx_train_new=torch.from_numpy(np.delete(np.array(idx_train), buf,0))
    return idx_train_new
        

def one_hot(x, class_count):
	return torch.eye(class_count)[x,:]


def jump_probability(mx, power):
    """calculate probability matrix"""
    rowsum = np.array(mx.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return np.dot(mx,mx)


class CEloss(nn.Module):
    def __init__(self):
        super(CEloss, self).__init__()

    def forward(self, y, y_onehot):
        # y: [N, C] logsoftmax
        # y_onehot: [N,C] one hot
        loss = torch.tensor(0,dtype=float,requires_grad=True)
        for i in range(len(y_onehot)):
            loss= loss - torch.sum(y_onehot[i]*y[i])
        return torch.div(loss,torch.tensor(len(y_onehot)))


class Sample_weighted_CEloss(nn.Module):
    def __init__(self):
        super(Sample_weighted_CEloss, self).__init__()

    def forward(self, y, y_onehot, weight):
        # y: [N, C] logsoftmax
        # y_onehot: [N,C] one hot
        loss = torch.tensor(0,dtype=float,requires_grad=True)
        for i in range(len(y_onehot)):
            loss= loss - torch.mul(weight[i],torch.sum(y_onehot[i]*y[i]))
        return torch.div(loss,torch.tensor(len(y_onehot)))

def set_file_name(args):
    result_dir=args.result_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    return result_dir+"/"


def print_fin_result(args,record, stage=0):
    bit_list = sorted(record.keys())
    bit_list.reverse()
    for key in bit_list[:1]:
        value = record[key]
        print('Stage:{:.0f}'.format(stage),'f1_val: {:.4f}'.format(key),'f1_test: {:.4f}'.format(value[1]))

    return

def activation_vec(activation, threshold):
    RF_num_buf=(activation>threshold).sum(1)
    RF_num=[]
    for i in RF_num_buf:
        RF_num.append(int(i))
    return RF_num

def activation_vec_all(activation, threshold):
    act_all=(activation>threshold).sum(0)
    RF_num= int((act_all>0).sum(1))
    return RF_num


def cal_entropy(RF):
    summ=sum(RF)
    if summ==0:
        result=0
    else:
        pro=np.array(RF)/summ
        result=0
        for x in pro:
            if x==0:
                x=1e-7
            result+=(-x)*math.log(x,2)

    return result/(-math.log(1/(len(RF)),2))

def board_log(args, record, final_record, opt):
    bit_list = sorted(record.keys())
    bit_list.reverse()
    key = bit_list[0]
    value = record[key]
    final_record[opt]=value    
    return final_record

def set_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    