import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Shrinking')

    # PATHS

    parser.add_argument('--results-path', type=str, default="./",
                        help="Where to save results (default: './results')")

    parser.add_argument('--checkpoint-path', type=str, default="./",
                        help="Where to save models (default: './checkpoints')")

    parser.add_argument('--dataset-path', type=str, default="./data",
                        help="Where to get the dataset (default: './dataset')")

    # TRAINING

    parser.add_argument('--batch-size', type=int, default=1,
                        help='Input batch size for training (default: 1)')

    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='Input batch size for testing (default: 1)')

    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Distributes the model across available GPUs.")

    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode: epochs are only performed on one randomly generated image")

    parser.add_argument('--device', type=str, default="cuda",
                        help="Device to use (default: 'cuda')")

    # HYPERPARAMETERS

    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')

    parser.add_argument('--wd', default="1e-4", type=float,
                        help='Weight decay rate (default: 1e-4)')

    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of epochs to train (default: 90)')

    parser.add_argument('--scheduler', type=str, default="multistep",
                        help="Scheduler to use (multistep, cosine or cosinewr)")

    parser.add_argument('--T-0', type=int, default=10,
                        help='For CosineAnnealingWarmRestarts: number of iterations for the first restart')

    parser.add_argument('--T-mult', type=int, default=2,
                        help='For CosineAnnealingWarmRestarts: a factor increases Ti after a restart')

    # RESNET OPTIONS

    parser.add_argument('--model', type=str, default="resnet50",
                        help="ResNet model to train (default: 'resnet50')")

    parser.add_argument('--resnet-width', type=int, default=64,
                        help='Input feature maps of classification ResNets')

    parser.add_argument('--first-layer-width', type=int, default=64,
                        help='Input feature maps of first layer of ResNets')

    return parser.parse_args()
