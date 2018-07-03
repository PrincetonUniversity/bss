import argparse
from bss.data import load_data
from bss.models.probit import Probit


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation on data files.')
    parser.add_argument('--file_pattern', dest='file_pattern', required=True)
    parser.add_argument('--iters', dest='iters', default=50, type=int)
    parser.add_argument('--burnin', dest='burnin', default=1000, type=int)

    return parser.parse_args()


def main():
    opts = parse_args()
    X, y, R = load_data(opts.file_pattern)
    model = Probit(
        X=X,
        Y=y,
        R=R
    )
    model.run_mcmc(burn_in=opts.burnin, iters=opts.iters)


if __name__ == '__main__':
    """
    Run the BSS Probit model on a given set of data files, with specified burn-in and iterations.
    
    To give this script a test run, use something like:
    cd </path/to/folder/containing/bss/package>
    python bss.scripts.run --file_pattern tests/sample_data/real0* --iters 10 --burnin 5
    """
    main()
