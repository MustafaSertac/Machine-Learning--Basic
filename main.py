import numpy as np
import argparse


def create_dataset(args, type, F):
    """
    Input(s):
        args (argparse.Namespace): parsed user arguments
        type (str): determines whether to create in sample or out sample instances
        F (ndarray): the f function we want to estimate
    
    Output(s):
        x (ndarray): an n-dimensional array of shape (N, d+1) which stores the features
        y (ndarray): an n-dimensional array of shape (N, 1) which stores the target values
    """

    # first we decide on the number of instances by checking the type
    if type == 'in':
        n = args.n_in
    elif type == 'out':
        n = args.n_out

    # then we randomly create features for the in/out sample dataset    
    x0 = np.ones(shape=(n, 1))
    x1 = np.random.uniform(low=args.low, high=args.high, size=(n, 1))
    x2 = np.random.uniform(low=args.low, high=args.high, size=(n, 1))
    x = np.hstack((x0, x1, x2))

    if args.model == 'nlt-reg':
        # if the model is non-linear transformation followed by regression, we create 
        # the custom F function and determine the ground truth target values
        F = np.multiply(x1, x1) + np.multiply(x2, x2) - 0.6
        y = F > 0
        y = y.astype(int)
        y = np.where(y < 1, -1, y)
        y = y.reshape(args.n_in, 1)

        # randomly introduce noise to 10% of data
        sim_noise = np.random.choice(np.arange(args.n_in), replace=False, size=int(args.n_in * 0.1))
        y[sim_noise] *= -1

        # check if the features should be also transformed
        if args.transform == 'True':
            x = np.hstack((x0, x1, x2, np.multiply(x1, x2), np.multiply(x1, x1), np.multiply(x2, x2)))

    else:
        # if the model is perceptron/regression, we create a random F function and determine
        # the ground truth target values
        det = (F[1][0] - F[0][0]) * (x2 - F[0][1]) - (F[1][1] - F[0][1]) * (x1 - F[0][0])
        y = det < 0
        y = y.astype(int)
        y = np.where(y < 1, -1, y)
        y = y.reshape(n, 1)

    # outputs
    return x, y

def predict(W, X, y):
    """
    Input(s):
        W (ndarray): an n-dimensional array of shape (d+1, 1) which stores the parameters
        X (ndarray): an n-dimensional array of shape (N, d+1) which stores the features
        y (ndarray): an n-dimensional array of shape (N, 1) which stores the target values
    
    Output(s):
        j (list): list of missclassified indices
    """

    # perform matrix multiplication and determine predictions (+1/-1)
    prod = np.matmul(W.T, X.T)
    check_pos = np.where(prod > 0, 1, prod)
    check_neg = np.where(prod < 0, -1, check_pos)

    # determine the missclassified indices
    miss_classified = check_neg != y.T
    i, j = np.where(miss_classified == True)

    # return the list of missclassified indices
    return j

def perceptron(args, X_in, y_in, X_out, y_out, W):
    """
    Input(s):
        args (argparse.Namespace): parsed user arguments
        X_in (ndarray): an n-dimensional array of shape (N, d+1) which stores the features of training data
        y_in (ndarray): an n-dimensional array of shape (N, 1) which stores the target values of training data
        X_out (ndarray): an n-dimensional array of shape (N, d+1) which stores the features of test data
        y_out (ndarray): an n-dimensional array of shape (N, 1) which stores the target values of test data
        W (ndarray): an n-dimensional array of shape (d+1, 1) which stores the parameters
    
    Output(s):
        total_iterations (list): total number of iterations which took pla to converge
        conflicts_in (list): in sample error
        conflicts_out (list): out sample error
    """

    total_iterations = 0
    conflict_in = []
    conflict_out = []

    # loop until the pla converges
    while True:
        # get the missclassified indices
        j = predict(W, X_in, y_in)

        # stop updating parameters if there are no more misclassifications
        if len(j) == 0:
            break

        # calculate e_in
        conflict_in.append(len(j) / args.n_in)

        # randomly select a misclassified index and update the parameters based on it
        miss_idx = np.random.choice(j)
        yx = (y_in[miss_idx] * X_in[miss_idx]).reshape(3, 1)
        W += yx

        total_iterations += 1

    # calculate e_out
    j = predict(W, X_out, y_out)
    conflict_out.append(len(j) / args.n_out)

    # return values
    return total_iterations, conflict_in, conflict_out

def regression(args, X_in, y_in, X_out, y_out):
    """
    Input(s):
        args (argparse.Namespace): parsed user arguments
        X_in (ndarray): an n-dimensional array of shape (N, d+1) which stores the features of training data
        y_in (ndarray): an n-dimensional array of shape (N, 1) which stores the target values of training data
        X_out (ndarray): an n-dimensional array of shape (N, d+1) which stores the features of test data
        y_out (ndarray): an n-dimensional array of shape (N, 1) which stores the target values of test data

    Output(s):
        total_iterations (list): total number of iterations which took pla to converge
        conflicts_in (list): in sample error
        conflicts_out (list): out sample error
        W (ndarray): an n-dimensional array of shape (d+1, 1) which stores the parameters
    """

    # calculate w based on this formula: w = (((X.T * X)^-1) * X.T) * y
    x_dag = np.matmul(np.linalg.inv(np.matmul(X_in.T, X_in)), X_in.T)
    W = np.matmul(x_dag, y_in)

    # calculate misclassified indices of training and test data
    j_in = predict(W, X_in, y_in)
    j_out = predict(W, X_out, y_out)

    total_iterations = 1
    conflicts_in = [len(j_in) / args.n_in]
    conflicts_out = [len(j_out) / args.n_out]

    # return out values
    return total_iterations, conflicts_in, conflicts_out, W

def main(args):
    """
    Input(s):
        args (argparse.Namespace): parsed user arguments

    Output(s):
        None
    """

    iterations = []
    total_conflict_in = []
    total_conflict_out = []

    # loop for each step
    for iter in range(args.total_iter):
        
        # create random F function and create datasets
        F = np.random.uniform(low=args.low, high=args.high, size=(2,2))
        X_in, y_in = create_dataset(args, 'in', F)
        X_out, y_out = create_dataset(args, 'out', F)
        W = np.zeros(shape=(3,1))

        # if user wants to do pla
        if args.model == 'perceptron':
            total_iterations, conflict_in, conflict_out = perceptron(args, X_in, y_in, X_out, y_out, W)

        # if user wants to do regression
        elif args.model == 'regression':
            total_iterations, conflict_in, conflict_out, W = regression(args, X_in, y_in, X_out, y_out)

        # if user wants to do regression and then pla
        elif args.model == 'reg-pla':
            total_iterations, conflict_in, conflict_out, W = regression(args, X_in, y_in, X_out, y_out)
            total_iterations, conflict_in, conflict_out = perceptron(args, X_in, y_in, X_out, y_out, W)

        # if user wants to do non-linear transformation and then regression
        elif args.model == 'nlt-reg':
            total_iterations, conflict_in, conflict_out, W = regression(args, X_in, y_in, X_out, y_out)

        # append the returned values
        iterations.append(total_iterations)
        total_conflict_in += conflict_in
        total_conflict_out += conflict_out

    # prints stats
    if args.transform == 'True': print('g(x1,x2) = ', W)
    print('average iterations: {}'.format(round(np.mean(iterations), 5)))
    print('average e_in: {}'.format(round(np.mean(total_conflict_in), 5)))
    print('average e_out: {}'.format(round(np.mean(total_conflict_out), 5)))

def parse_args():
    parser = argparse.ArgumentParser("Solutions to Homework 2")
    parser.add_argument('-n_in', type=int, default=10, help='total number of in sample instances')
    parser.add_argument('-n_out', type=int, default=1000, help='total number of out of sample instances')
    parser.add_argument('-low', type=int, default=-1, help='min. bound of instances')
    parser.add_argument('-high', type=int, default=1, help='max. bound of instances')
    parser.add_argument('-total_iter', type=int, default=1000, help='total iterations')
    parser.add_argument('-model', type=str, default='perceptron', help='the linear model')
    parser.add_argument('-transform', type=str, default='False', help='linear transformation')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()    
    main(args)
