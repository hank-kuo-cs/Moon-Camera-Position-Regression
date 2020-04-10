import sys
from Regression import train, test, validate

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('You have to assign regression mode! Like command below:' +
                         '\npython run_regression.py <mode>' +
                         '\n<mode>: train, test or validate')

    mode = sys.argv[1]

    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    elif mode == 'validate':
        validate()
    else:
        raise ValueError('Mode must be "train", "test" or "validate", but get"%s"' % mode)
