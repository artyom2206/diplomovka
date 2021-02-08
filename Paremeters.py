from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.callbacks import CSVLogger


TARGET_UPDATE_PERIOD = 1000

LEARNING_RATE = 0.00025

DISCOUNT_FACTOR = 0.99

BATCH_SIZE = 32

NOOPMAX = 10

MAX_EXPERIENCES = 1000000  # Memory Size 40k

MIN_EPSILON = 0.05

DECAY_RATE = 0.99999

TOTAL_LIVES = 3

ADAM_Opt = Adam(lr=LEARNING_RATE)

EXPLORATION_TEST = 0.1
