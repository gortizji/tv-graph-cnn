import numpy as np

from applications.seirs.data_utils import create_train_test_mb_sources, DATASET_FILE

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

TRAIN_SIZE = 4000
TEST_SIZE = 1000

if __name__ == '__main__':
    train_mb_source, test_mb_source = create_train_test_mb_sources(DATASET_FILE, TEST_SIZE/(TRAIN_SIZE+TEST_SIZE), n_samples=TEST_SIZE + TRAIN_SIZE)

    batch, end = train_mb_source.next_batch(TRAIN_SIZE)

    Xtrain = np.reshape(batch[0], [batch[0].shape[0], -1])
    ytrain = batch[1]

    batch, end = test_mb_source.next_batch(TEST_SIZE)

    Xtest = np.reshape(batch[0], [batch[0].shape[0], -1])
    ytest = batch[1]

    # lr = LinearRegression()
    # lr.fit(Xtrain, ytrain)
    # yest = lr.predict(Xtest)
    #
    # r2_lr = r2_score(ytest, yest)
    #
    # print("R2 of Linear Regression: ", r2_lr)
    #

    # dt = DecisionTreeRegressor(max_depth=10)
    # dt.fit(Xtrain, ytrain)
    # yest = dt.predict(Xtest)
    #
    # r2_dt = r2_score(ytest, yest)
    #
    # print("R2 of Decision Tree Regressor: ", r2_dt)

    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(Xtrain, ytrain)
    yest = rf.predict(Xtest)

    r2_rf = r2_score(ytest, yest)

    print("R2 of Random Forest: ", r2_rf)