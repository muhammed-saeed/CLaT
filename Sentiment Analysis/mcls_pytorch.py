# Importing the libraries needed
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import RobertaModel, RobertaTokenizer
import random
random.seed(1)

tokenizer_folder = "PATH_TO/tokenizer"
#when to the tokenizer


from transformers import RobertaTokenizerFast, BertTokenizerFast, BertTokenizer
# Create the tokenizer from a trained one
MAX_LEN = 512
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# tokenizer = BertTokenizer.from_pretrained(tokenizer_folder,max_len=MAX_LEN)
#incase using the bertTokenizer from scratch





# Setting up the device for GPU usage

from torch import cuda
device = 'cuda:3' if cuda.is_available() else 'cpu'


train_path = "PATH_TO/pidgin/pcm_train.csv"
test_path = "PATH_TO/pidgin/pcm_test.csv"
dev_path = "PATH_TO/pidgin/pcm_dev.csv"

df_train = pd.read_csv(train_path)
df_dev = pd.read_csv(dev_path)
frames= [df_train, df_dev]
df = pd.concat(frames)
df_test = pd.read_csv(test_path)

df = df[['text','label']]
df_test = df_test[['text', 'label']]
# df.head()

# # Converting the codes to appropriate categories using a dictionary
my_dict = {"positive":2, "neutral":1, "negative":0 }


def update_cat(x):
    return my_dict[x]

df['label'] = df['label'].apply(lambda x: update_cat(x))
df_test['label'] = df_test['label'].apply(lambda x: update_cat(x))

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['label'].apply(lambda x: encode_cat(x))
df_test['ENCODE_CAT'] = df_test['label'].apply(lambda x: encode_cat(x))

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 1e-05
from transformers import BertForMaskedLM, BertTokenizer,BertModel

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model =  BertForMaskedLM.from_pretrained("bert-base-uncased")



class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.text[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len




train_dataset = df.reset_index(drop=True)
test_dataset = df_test.reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(df.shape))
print("TEST Dataset: {}".format(df_test.shape))

training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

print(training_set)
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


model_path = "PATH_TO_CHECKPOINT"

# model = BertModel.from_pretrained("bert-base-uncased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 
class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        # self.l1 = RobertaModel.from_pretrained("roberta-base")
        
        self.l1 = BertModel.from_pretrained(model_path)

        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return 
for epoch in range(EPOCHS):
    train(epoch)


from sklearn.metrics import f1_score
target_ = []
output_ = []
idx_ = []
print("Complete Train and Going into the valid")
def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps = 0
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            output_.extend(outputs.cpu().detach().numpy().tolist())
            loss = loss_function(outputs, targets)
            target_.extend(targets.cpu().detach().numpy().tolist())
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            idx_.extend(big_idx.cpu().detach().numpy().tolist())
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu
print('This is the validation section to print the accuracy and see how it performs')
print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')

acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)

from sklearn import metrics

numpy_idx = idx_
numpy_target = target_

accuracy = metrics.accuracy_score(target_, idx_)

print(f"accuracy score using output is {accuracy}")
f1_score_weighted = metrics.f1_score(target_, idx_, average='weighted')
print(f"f1 score using output is {f1_score_weighted}")

