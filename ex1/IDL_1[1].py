"""
# **1)**
"""

import torch.utils.data as data
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class HLADataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.data_input = []
        self.data_label = []

        types = "ACDEFGHIKLMNPQRSTVWY"
        self.type_to_index = {}

        for i in range(__builtins__.len(types)):  # Explicitly using __builtins__.len()
            self.type_to_index[types[i]] = i

        self.mapping_file = {
            'negs.txt': 0,
            'A0101_pos.txt': 1,
            'A0201_pos.txt': 2,
            'A0203_pos.txt': 3,
            'A0207_pos.txt': 4,
            'A0301_pos.txt': 5,
            'A2402_pos.txt': 6
        }

        for input_file, label in self.mapping_file.items():
            with open(input_file, "r") as f:
                for line in f:
                    line = line.strip()

                    if __builtins__.len(line) == 9:  # Explicitly using __builtins__.len()
                        self.data_input.append(line)
                        self.data_label.append(label)

    def one_hot_encode(self, sequence):
        vector = torch.zeros(9, 20)

        for i in range(9):
            vector[i][self.type_to_index[sequence[i]]] = 1

        return vector.flatten()

    def __len__(self):
        return __builtins__.len(self.data_input)  # Explicitly using __builtins__.len()

    def __getitem__(self, idx):
        return self.one_hot_encode(self.data_input[idx]), self.data_label[idx]


"""so now to do instanse of the data and split it for test loadre and train loader:"""

dataset = HLADataset()
lengeth_set = dataset.__len__()

train_len = int(lengeth_set * 0.9)
test_len = lengeth_set - train_len
trainset, testset = data.random_split(dataset, [train_len, test_len])

train_loader = data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = data.DataLoader(testset, batch_size=128, shuffle=False)

"""
## **2)b**

"""


class MyMLP(nn.Module):

    def __init__(self):
        super().__init__()
        # Initialize the modules we need to build the network
        self.my_module = nn.Sequential(
            nn.Linear(180, 180),
            nn.ReLU(),
            nn.Linear(180, 180),
            nn.ReLU(),
            nn.Linear(180, 7)
        )

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        return self.my_module(x)


"""train, test the model and plot the results:"""


def train_model(model, data_loader,
                test_loader, loss_function,
                optimizer, device, num_epochs=20):
    train_looses = []
    test_looses = []

    for epoch in tqdm(range(num_epochs)):
        ## training phase
        model.train()
        curr_traind_loss = []
        for data_input, data_label in data_loader:
            data_input = data_input.to(device)
            data_label = data_label.to(device)

            prediction = model(data_input)
            loss = loss_function(prediction, data_label)
            curr_traind_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        avg_train_loss = sum(curr_traind_loss) / __builtins__.len(curr_traind_loss)
        train_looses.append(avg_train_loss)

        ## evaluation phase
        model.eval()
        curr_test_loss = []
        with torch.no_grad():
            for data_input, data_label in test_loader:
                data_input = data_input.to(device)
                data_label = data_label.to(device)

                prediction = model(data_input)
                loss = loss_function(prediction, data_label)
                curr_test_loss.append(loss.item())

        avg_test_loss = sum(curr_test_loss) / __builtins__.len(curr_test_loss)
        test_looses.append(avg_test_loss)

        print(
            f"Epoch: [{epoch + 1}/{num_epochs}] | Training loss: {avg_train_loss:.4f} | Test loss: {avg_test_loss:.4f}")

    plt.plot(train_looses, label="train loss")
    plt.plot(test_looses, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss over each epoch")
    plt.legend()
    plt.show()


"""define the MLP:"""

my_first_model = MyMLP()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
my_first_model.to(device)

wights_err = torch.tensor([1.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]).to(device)
train_model(my_first_model, train_loader,
            test_loader, nn.CrossEntropyLoss(weight=wights_err),
            torch.optim.SGD(my_first_model.parameters(), lr=0.1),
            device)

"""
## **2)c**
"""



class MyExpandedMLP(nn.Module):

    def __init__(self):
        super().__init__()
        # Initialize the modules we need to build the network
        self.my_module = nn.Sequential(
            nn.Linear(180, 90),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(90, 45),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(45, 7)
        )

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        return self.my_module(x)


my_expanded_model = MyExpandedMLP()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
my_expanded_model.to(device)

wights_err = torch.tensor([1.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]).to(device)
train_model(my_expanded_model, train_loader,
            test_loader, nn.CrossEntropyLoss(weight=wights_err),
            torch.optim.Adam(my_expanded_model.parameters(), lr=0.001, weight_decay=1e-4),
            device)

"""## **2)d**
We have to remove the activation functions from the network(the relu between the layers) so we will get :
"""


class MyLinearMLP(nn.Module):

    def __init__(self):
        super().__init__()
        # Initialize the modules we need to build the network
        self.my_module = nn.Sequential(
            nn.Linear(180, 90),
            nn.Dropout(0.4),

            nn.Linear(90, 45),
            nn.Dropout(0.4),

            nn.Linear(45, 7)
        )

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        return self.my_module(x)


my_linear_model = MyLinearMLP()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
my_linear_model.to(device)

wights_err = torch.tensor([1.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]).to(device)
train_model(my_linear_model, train_loader,
            test_loader, nn.CrossEntropyLoss(weight=wights_err),
            torch.optim.Adam(my_linear_model.parameters(), lr=0.001, weight_decay=1e-4),
            device)

"""# **2)e**
"""

my_expanded_model.eval()

spike_protein = """
MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFS
NVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIV
NNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLE
GKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQT
LLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETK
CTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISN
CVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIAD
YNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPC
NGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVN
FNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITP
GTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSY
ECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTI
SVTTEILPVSMTKTSVDCVMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQE
VFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDC
LGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAM
QMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALN
TLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRA
SANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPA
ICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDP
LQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDL
QELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDD
SEPVLKGVKLHYT
""".replace("\n", "").replace(" ", "").strip()

len_spike = len(spike_protein)

spike_results = []
alleles = {1: 'A0101',
           2: 'A0201',
           3: 'A0203',
           4: 'A0207',
           5: 'A0301',
           6: 'A2402'}

with torch.no_grad():
    for i in range(len_spike - 8):
        peptide = spike_protein[i:i + 9]

        x = dataset.one_hot_encode(peptide).unsqueeze(0).to(device)

        outputs = my_expanded_model(x)
        probs = torch.softmax(outputs, dim=1)

        prob, number_class = torch.max(probs[0][1:], dim=0)

        spike_results.append({
            'peptide': peptide,
            'probability': prob.item(),
            'allele': alleles[number_class.item() + 1]
        })

spike_results.sort(key=lambda x: x['probability'], reverse=True)

for i in range(3):
    res = spike_results[i]
    print(f"Rank {i + 1}: {res['peptide']} | Allele: {res['allele']} | Score: {res['probability']:.4f}")
