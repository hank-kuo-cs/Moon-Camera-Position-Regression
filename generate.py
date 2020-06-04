import sys
from Regression.generate import generate_dataset, generate_metric_dataset


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'metric':
        generate_metric_dataset()
    elif len(sys.argv) == 1:
        generate_dataset()
    else:
        raise ValueError('Cannot understand the command <%s>' % sys.argv[1])
