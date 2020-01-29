import numpy as np

class Dataset(object):
    def __init__(self, path):
        self.path = path

        #load the dataset and get the dimensions
        self.dataset, self.dims, self.num_context = self.read_data(path + "/ratings.dat")

        # compute the global mean
        self.global_mean = self.get_global_mean()

        self.num_users, self.num_items = self.dims[0], self.dims[1]
        self.num_ratings = len(self.dataset[1])

        self.training_set, self.validation_set, self.testing_set = self.training_validation_testing()

    def training_validation_testing(self):
        training_set, training_dims, training_num_context = self.read_data(self.path + "/ratings_train.txt")
        validation_set, validation_dims, validation_num_context = self.read_data(self.path + "/ratings_valid.txt")
        testing_set, testing_dims, testing_num_context = self.read_data(self.path + "/ratings_test.txt")
        return training_set, validation_set, testing_set

    def read_data(self, file_name):
        X = np.loadtxt(file_name, dtype=float, delimiter=',')
        ndims = X.shape[1] - 1
        Y = X.T[ndims]  # rating values
        X = np.delete(X, ndims, 1).astype(int)  # index values
        dims = [len(list(set(X.T[i]))) for i in range(ndims)]
        context_set = set()
        for dim in range(ndims):
            if dim > 1:
                context_set = context_set | set(X.T[dim])
        num_context = len(context_set)
        return [X, Y], dims, num_context

    def get_global_mean(self):
        sum = 0
        for entry in self.dataset[1]:
            sum = sum + entry
        global_mean = sum/len(self.dataset[1])
        return global_mean