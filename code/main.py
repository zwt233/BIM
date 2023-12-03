from pymodule import *
from utils import *
from network import *


def train(model, optimizer, record,args, model_name):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, flag=0)
    if args.hard_label==True:
        loss_CEloss=Sample_weighted_CEloss()
        if args.cuda==True:
            loss_CEloss=loss_CEloss.cuda()
        loss_train = loss_CEloss(output[idx_train], labels_onehot[idx_train],loss_weight[idx_train])
    else:  
        loss_CEloss=CEloss()
        if args.cuda==True:
            loss_CEloss=loss_CEloss.cuda()
        loss_train = loss_CEloss(output[idx_train], labels_softonehot[idx_train])
    loss_train.backward()
    optimizer.step()
    
    with torch.no_grad(): 
        model.eval()
        output = model(features, adj, flag=1)
    f1_val = f1_score(labels[idx_val].cpu(),torch.max(output[idx_val,:],1)[1].cpu(), average="macro")

    if f1_val > args.max_f1_val:
        args.max_f1_val = f1_val

        torch.save(model.state_dict(),model_name)
        f1_test = f1_score(labels[idx_test].cpu(),torch.max(output[idx_test,:],1)[1].cpu(), average="macro") 
        acc_test = accuracy(output[idx_test].cpu(), labels[idx_test].cpu())
        auc_test=roc_auc_score(labels[idx_test].cpu(),output[idx_test,:].cpu().detach().numpy(), average="macro",multi_class='ovo')

        
        record[f1_val.item()] = [f1_val.item(), f1_test.item(), acc_test.item(), auc_test.item()]

def RF_calculation():
    idx_nodes={}
    activation={}
    for i in range(class_num):
        idx_nodes[i]=[]
        activation[i]=[]
    for i in idx_train_round_0:
        idx_nodes[int(labels[i])].append(int(i))
    for i in range(class_num):
        if i==0:
            activation=adj_probability[idx_nodes[i],:].sum(axis=0)
        else:
            activation=np.row_stack((activation, adj_probability[idx_nodes[i],:].sum(axis=0)))

    # Add new nodes
    if idx_train_buffer.size!=0:
        for idx in idx_train_buffer:
            activation=activation+(confidence_score[idx,:].reshape(-1,1))*adj_probability[idx,:]

    RF_num=activation_vec(activation, args.T_influence)
    RF_num_all=activation_vec_all(activation, args.T_influence)
    return RF_num,RF_num_all, activation

def search_nodes():
    dif=[]
    dif_class=[]   
    activation_buf={}
    RF_num_buf={}
    RF_num_all_buf={}
    class_idx=[]
    normal_entropy={}
    RF_para_buf={}
    idx_search=[]

    for i in idx_unlabeled:
        con_score=confidence_score[i].max(0)
        if con_score>=args.T_confidence:
            idx_search.append(i)


    for i in range(len(idx_search)):
        activation_buf[i]=[]
        idx=idx_search[i]
        class_idx.append(np.argmax(confidence_score[idx]))
        if  nodes[class_idx[i]]<args.max_nodes_num:
            activation_buf[i]=activation+(confidence_score[idx].reshape(-1,1)*adj_probability[idx,:])
            RF_num_buf[i]=activation_vec(activation_buf[i], args.T_influence)
            RF_num_all_buf[i]=activation_vec_all(activation_buf[i], args.T_influence)
            normal_entropy[i] = cal_entropy(RF_num_buf[i])
            RF_para_buf[i]=(RF_num_all_buf[i]/labels.shape[0])+args.alpha*normal_entropy[i]
            if (RF_num_buf[i]==RF_num and RF_num_all_buf[i]-RF_num_all==0) or (RF_para_buf[i]<=RF_para):
                activation_buf[i]=-1
                RF_num_buf[i]=-1
                RF_num_all_buf[i]=-1
                dif.append(-1000000)
            else:
                dif_class.append(i)
                dif.append(RF_para_buf[i])
        else:
            activation_buf[i]=-1
            RF_num_buf[i]=-1
            RF_num_all_buf[i]=-1
            dif.append(-1000000)
    return np.array(dif), activation_buf, np.array(dif_class).astype(int), RF_num_buf, RF_num_all_buf, class_idx, normal_entropy, RF_para_buf, idx_search

def best_node_cal():

    max_dif=np.max(dif[dif_class])
    dif_best_idx=np.argmax(dif)
    pseudo_node[class_idx[dif_best_idx]]+=1
    nodes[class_idx[dif_best_idx]]+=1

    
    best_idx=idx_search[dif_best_idx]
    global idx_unlabeled
    del_dif_idx=np.where(idx_unlabeled==best_idx)[0]

    # Select the max RF increase node
    global idx_train_buffer
    idx_train_buffer=np.append(idx_train_buffer,best_idx).astype(np.int64)

    # # Hard label
    ones=np.ones([class_num])
    ones[np.argmax(confidence_score[best_idx])]=0
    labels_onehot[best_idx,:]=torch.tensor((1-ones))
    labels[best_idx]=np.argmax(confidence_score[best_idx])
    loss_weight[best_idx]=torch.tensor(confidence_score[best_idx,np.argmax(confidence_score[best_idx])])
    # # Soft label
    labels_softonehot[best_idx,:]=torch.tensor(confidence_score[best_idx,:])

    activation=activation_buf[dif_best_idx]
    RF_num=RF_num_buf[dif_best_idx]
    RF_num_all=RF_num_all_buf[dif_best_idx]
    idx_unlabeled=np.delete(idx_unlabeled, del_dif_idx) 
    entropy = normal_entropy[dif_best_idx]
    RF_para = RF_para_buf[dif_best_idx]
    return RF_num, RF_num_all, activation, entropy, RF_para

def set_cuda():
    global model, features, adj, labels, idx_test, idx_train, idx_val, idx_train_original, labels_onehot, loss_weight
    model = model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    labels_onehot = labels_onehot.cuda()
    loss_weight= loss_weight.cuda()
    idx_train_original=idx_train_original.cuda()

def set_cpu():
    global labels, labels_onehot, loss_weight, idx_train 
    labels = labels.cpu()
    labels_onehot=labels_onehot.cpu()
    idx_train = idx_train.cpu()
    loss_weight= loss_weight.cpu()  
    


#######################################################################################################

parser = get_parser()
args = parser.parse_args()

args.result_dir='result_'+str(args.dataset)
args.train_initial_model=True
print("\nResult dir:", args.result_dir)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('Cuda:', args.cuda)
result_dir=set_file_name(args)
csv_file=result_dir+'result_'+args.dataset+'.csv'

with open(csv_file,'w+',newline='')as f:
    fieldnames = {'f1_test','acc_test','auc_test'}
    writer = csv.DictWriter(f,fieldnames=fieldnames)
    writer.writeheader()

para_list={}
para_result={}
time_begin=time.time() 

# Dataloader
adj, features, labels, labels_onehot, idx_train, idx_val, idx_test, idx_unlabeled, adj_original= load_data(args=args,path="../data")
sample_num=labels_onehot.size()[0]
class_num=labels_onehot.size()[1]

idx_train_original=idx_train.clone()
labels_softonehot=labels_onehot.clone()
loss_weight=torch.ones(sample_num)
final_record={}
nodes=np.zeros([class_num]) 
for i in range(class_num):
    nodes[i]=sum(labels_onehot[idx_train][:,i]) 

# # Train stage 0
#Model and optimizer
set_seed(args.seed)
model = GCN(nfeat=features.shape[1],nhid=args.nhid,nclass=labels.max().item() +1,dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

args.max_f1_val=0.1
if args.train_initial_model==True:
    record = {}
    if args.cuda==True:
        set_cuda()
    for epoch in range(args.epochs):
        train(model,optimizer,record,args,model_name=result_dir+"model_0.pth")
    if args.cuda==True:
        set_cpu()        
    print_fin_result(args,record,stage=0)
    final_record = board_log(args, record, final_record, opt=0)
    model.load_state_dict(torch.load(result_dir+"model_0.pth",map_location='cpu'))
else:     
    model.load_state_dict(torch.load("model_0.pth",map_location='cpu'))
    with torch.no_grad():
        model.eval()
        output = model(features, adj, flag=1)
    f1_val = f1_score(labels[idx_val],torch.max(output[idx_val,:],1)[1], average="macro")
    f1_test = f1_score(labels[idx_test],torch.max(output[idx_test,:],1)[1], average="macro")
    print('Stage:0','f1_val: {:.4f}'.format(f1_val),'f1_test: {:.4f}'.format(f1_test))
    final_record[0]=[f1_val.item(), f1_test.item()]
 

# # Pseudo label learning
# Probability matrix
adj_probability=jump_probability(adj_original+sp.eye(adj_original.shape[0]),power=2)
idx_train_round_0=idx_train.clone()
idx_train_buffer=np.array([])

# Initail RF calculation

with torch.no_grad():
    model.eval()
    confi= model(features, adj, flag=1)
    confidence=confi.cpu()
confidence_score=np.array(confidence.clone())
RF_num_0, RF_num_all_0, activation = RF_calculation() 
entropy_0 = cal_entropy(RF_num_0)
RF_para_0=RF_num_all_0/labels.shape[0]+args.alpha*entropy_0


pseudo_node=np.zeros([class_num]) 
RF_num=RF_num_0
RF_num_all=RF_num_all_0
RF_para=RF_para_0
while 1:
    dif, activation_buf, dif_class, RF_num_buf, RF_num_all_buf, class_idx, normal_entropy, RF_para_buf, idx_search = search_nodes()
    if dif_class.size>0:
        RF_num, RF_num_all, activation, entropy, RF_para = best_node_cal()
    else:
        break

idx_train=torch.tensor(np.append(idx_train_round_0, torch.tensor(idx_train_buffer,dtype=int)))

# # Train stage 1
record = {}
args.max_f1_val=0.1
model_name=result_dir+"model_1.pth"
if args.train_new_model==True:
    set_seed(args.seed)
    model = GCN(nfeat=features.shape[1],nhid=args.nhid,nclass=labels.max().item() +1,dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
if args.cuda==True:
    set_cuda()
for epoch in range(args.epochs):
    train(model, optimizer, record, args, model_name=model_name)
if args.cuda==True:
    set_cpu()
print_fin_result(args,record,stage=1)
final_record = board_log(args, record, final_record, opt=1)



# # Train iteratively
confidence={}
for opt_num in range(args.optimize_num-1):
    confidence[opt_num]=[]
    # Initail RF calculation
    model_name=result_dir+"model_"+str(opt_num+1)+".pth"
    model.load_state_dict(torch.load(model_name))
    if args.cuda==True:
        model.cuda()
    with torch.no_grad():
        model.eval()
        confi= model(features, adj, flag=1)
        confidence[opt_num]=confi.cpu()

    for i in range(opt_num+1):
        if i==0:
            confidence_score=np.array(confidence[i].clone())
        else:
            confidence_score=confidence_score+np.array(confidence[i].clone())
    confidence_score=confidence_score/(opt_num+1)

    RF_num_0, RF_num_all_0, activation = RF_calculation()
    entropy_0 = cal_entropy(RF_num_0)
    RF_para_0=RF_num_all_0/labels.shape[0]+args.alpha*entropy_0 
    for change_idx in idx_train_buffer:
        loss_weight[change_idx]=torch.tensor(confidence_score[change_idx,np.argmax(confidence_score[change_idx])]) 

    RF_num=RF_num_0
    RF_num_all=RF_num_all_0
    RF_para=RF_para_0
    pseudo_node=np.zeros([class_num]) 
    while 1:
        dif, activation_buf, dif_class, RF_num_buf, RF_num_all_buf, class_idx, normal_entropy, RF_para_buf, idx_search = search_nodes()
        if len(dif_class)>0:
            RF_num, RF_num_all, activation, entropy, RF_para = best_node_cal()
        else:
            break

    idx_train=torch.tensor(np.append(idx_train_round_0, torch.tensor(idx_train_buffer,dtype=int)))

    # Training
    record = {}
    
    model_name=result_dir+"model_"+str(opt_num+2)+".pth"
    args.max_f1_val=0.1
    if args.train_new_model==True:
        set_seed(args.seed)
        model = GCN(nfeat=features.shape[1],nhid=args.nhid,nclass=labels.max().item() +1,dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda==True:
        set_cuda()        
    for epoch in range(args.epochs):
        train(model, optimizer, record, args, model_name=model_name)
    if args.cuda==True:
        set_cpu()
    print_fin_result(args,record,stage=opt_num+2)
    final_record = board_log(args, record, final_record, opt=opt_num+2)
val_score=[]
for i in range(len(final_record)):
    val_score.append(final_record[i][0])
best_stage = np.argmax(val_score)

write_record=  {'f1_test': round(final_record[best_stage][1],3), 'acc_test':round(final_record[best_stage][2],3), 'auc_test':round(final_record[best_stage][3],3)}

with open(csv_file,'a+',newline='')as f:
    fieldnames = {'f1_test','acc_test','auc_test'}
    writer = csv.DictWriter(f,fieldnames=fieldnames)
    writer.writerow(write_record)