import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

MARGIN = 0.2
EXPERIMENT_NAME = 'metric_resnet18_b16_sgd_lr1e3'
LEARNING_RATE = 0.001
EPOCH_NUM = 300
BATCH_SIZE = 16
DEVICE = 'cuda'
