from DataGenerator.DataGenerator import get_local_train_dataset
import hyper

get_local_train_dataset(0, hyper.path_train, hyper.category_features, hyper.skew_features, 0)