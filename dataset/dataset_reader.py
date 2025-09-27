import json
import os

class DatasetReader:
    """
    A class to read and process the dataset from a JSON file.
    """
    def __init__(self, sequence_dir):
        """
        Initializes the DatasetReader with the path to the JSON file.
        """
        self.sequence_dir = sequence_dir
        self.sequence_name = sequence_dir.split('/')[-1]
        self.json_path = os.path.join(sequence_dir, 'tasks.json')
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads the data from the JSON file.

        Returns:
            list: A list of dictionaries, where each dictionary represents a task.
        """
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset by its index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: The dataset item at the specified index.
        """
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer.")
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range.")
        return self.data[idx]

if __name__ == '__main__':

    sequence_dir = 'Medium-M2'

    dataset_reader = DatasetReader(
        sequence_dir=f"Dataset/images/{sequence_dir}"
    )

    # print(len(dataset_reader))
    # print(dataset_reader[0])
    # import pdb; pdb.set_trace()