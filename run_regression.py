import sys
from Regression import train, test, validate

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('You have to assign regression mode! Like command below:' +
                         '\npython run_regression.py <mode>' +
                         '\n<mode>: train, test or validate')

    if sys.argv[1] not in ['train', 'test', 'validate']:
        raise ValueError('Mode must be "train", "test" or "validate"')
