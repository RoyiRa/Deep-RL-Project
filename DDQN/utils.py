import itertools
from tensorflow.keras.optimizers import Adam
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prioritized Experience Replay vars
MEMORY_CAPACITY = 200000

# NN vars
#TODO: try reward cipping + gradient and norm clipping
optimizer = Adam(learning_rate=0.0001)#, clipnorm=1., clipvalue=0.5)
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)

# Quantization vars
m_engine = [-1., 0.2, 0.6]
s_engine = [-0.57, 0., 0.57]
all_discrete_actions = list(itertools.product(*[m_engine, s_engine]))
bucket_2_action = {bucket: idx for idx, bucket in enumerate(all_discrete_actions)}