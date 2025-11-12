import torch
import math
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from lib import *

# Ignore warning messages
warnings.filterwarnings('ignore')

# ==================== Experiment Configuration ====================
# Hyperparameter settings
LEARNING_RATE = 0.0001
REGULARIZATION_COEFFICIENT = 1e-3
NUM_FUZZY_SETS = 3
MAX_EPOCHS = 100
BATCH_SIZE_PROPORTION = 0.25

# load dataset
dataset_name = 'SRBCT'
dataset = torch.load(r'datasets/{}.pt'.format(dataset_name), weights_only=False)
sample, target = dataset.sample, dataset.target

# one-hot the label
target = torch.LongTensor(preprocessing.OneHotEncoder().fit_transform(target).toarray())

# split train-test samples
tra_sam, test_sam, tra_tar, test_tar = train_test_split(sample, target, train_size=0.7)

# preprocessing, linearly normalize the training and test samples into the interval [0,1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
tra_sam = torch.Tensor(min_max_scaler.fit_transform(tra_sam))
test_sam = torch.Tensor(min_max_scaler.transform(test_sam))

# No. samples, features, and classes
num_tra_sam, num_fea = tra_sam.shape
num_class = tra_tar.shape[1]

# init the model
myTSK = FirstTSK(num_fea, num_class, NUM_FUZZY_SETS, mf='GCGMF', tnorm='atp-softmin')

# training and test
batch_size = math.ceil(num_tra_sam * BATCH_SIZE_PROPORTION)
train_mini_batch(tra_sam, myTSK, tra_tar, LEARNING_RATE, MAX_EPOCHS,
                 batch_size=batch_size, optim_type='Adam', beta=REGULARIZATION_COEFFICIENT, verbose=True)

tra_loss, tra_acc = test(tra_sam, myTSK, tra_tar)
test_loss, test_acc = test(test_sam, myTSK, test_tar)
print(fr'{dataset_name} dataset, training acc: {tra_acc:.4f}, test acc: {test_acc:.4f}')