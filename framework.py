import utils


class Framework():

    def __init__(self):
        self.__datasets = utils.get_all_datasets_full_path()
        self.__counter = 0

    def start(self):
        for ds in self.__datasets:
            self.__counter += 1
            # split to train_test
            X, y = utils.split_datasets_to_x_y(ds)
            X_train, X_test, y_train, y_test = utils.split_to_train_test(X, y)
            self.fit_and_predict(X_train, X_test, y_train, y_test)
        print('yallaaaaaaaaaaaaa')


    def fit_and_predict(self, X_train, X_test, y_train, y_test):
        print(self.__counter)


if __name__ == '__main__':
    frm = Framework()
    frm.start()