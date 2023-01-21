import pandas as pd
from sklearn.model_selection import train_test_split

# mention the path of data file
data_file_path = r""
data_file = "\debateforum_gayrights"
data_file_type = ".csv"
train_data_file = data_file_path + data_file + r"_train" + data_file_type
test_data_file = data_file_path + data_file + r"_test" + data_file_type

dataset_path = data_file_path + data_file + data_file_type
print("\nReading data from path:\n{}".format(dataset_path))
dataset = pd.read_csv(dataset_path)
dataset = dataset.sample(frac=1)

print("\nFew instances from the dataset")
print(dataset.head())

train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=0)

print("\nNo. of instances in Train Dataset: {}".format(train_data.shape[0]))
print("\nWriting Train Data to file:\n{}".format(train_data_file))
train_data.to_csv(train_data_file, index=False)

print("\nNo. of instances in Test Dataset: {}".format(test_data.shape[0]))
print("\nWriting Test Data to file:\n{}".format(test_data_file))
test_data.to_csv(test_data_file, index=False)
