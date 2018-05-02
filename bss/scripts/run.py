import argparse
from bss.utils.data import load_data
from bss.models.probit import ProbitSS


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation on data files.')
    parser.add_argument('--file_pattern', dest='file_pattern', required=True)
    parser.add_argument('--iters', dest='iters', default=50, type=int)
    parser.add_argument('--burnin', dest='burnin', default=1000, type=int)

    return parser.parse_args()


def main():
    opts = parse_args()
    X, y, _, R = load_data(opts.file_pattern)
    model = ProbitSS(
        X=X,
        Y=y,
        R=R,
        sample_xi=False
    )
    model.run_mcmc(burnin=opts.burnin, iters=opts.iters)


if __name__ == '__main__':
    main()
