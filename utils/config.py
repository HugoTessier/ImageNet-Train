import os

# Dataset root

traindir = os.getenv("DATASET_TRAIN")
valdir = os.getenv("DATASET_VAL")

# Optimizer
lr = 0.5
momentum = 0.9
nesterov = False

weight_decay = 2e-05
norm_weight_decay = 0.0
label_smoothing = 0.1

# Scheduler
lr_warmup_decay = 0.01
epochs = 600
lr_warmup_epochs = 5

# DataLoaders
workers = os.cpu_count()

# EMA
model_ema_steps = 32
model_ema_decay = 0.99998

# Data augmentation
mixup_alpha = 0.2
cutmix_alpha = 1.0
ra_reps = 4
random_erase = 0.1
hflip_prob = 0.5

# Batch sizes
theoretical_batch_size_for_training = 1024
train_batch_size = 128
test_batch_size = 128

# Image sizes
test_image_size = (1, 3, 224, 224)
val_resize_size = 232
val_crop_size = 224
train_crop_size = 176

# Acceleration
channels_last = True
autotuner = True
pin_memory = True
persistent_workers = True
save_every_n_epochs = 10

# Miscellaneous
debug = False
results_path = './results'
checkpoint_path = './checkpoint'
