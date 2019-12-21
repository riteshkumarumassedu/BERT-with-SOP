import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import csv


class data_parser(Dataset):
    """ Dataset Class to read data from the  CSV file """
    labels = None
    def __init__(self, file, pipeline=[]):
        Dataset.__init__(self)
        data = []
        with open(file, "r") as f:
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance in self.get_instances(lines):
                for proc in pipeline:
                    instance = proc(instance)
                data.append(instance)

        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError


class load_MRPC_data(data_parser):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[3], line[4] # label, text_a, text_b


class load_MNLI_data(data_parser):
    """ Dataset class for MNLI """
    labels = ("contradiction", "entailment", "neutral") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[8], line[9] # label, text_a, text_b

class load_STSB_data(data_parser):
    """ Dataset Class for STSB"""
    labels = (None) # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[7], line[8] # label, text_a, text_b


class load_QQP_data(data_parser):
    """ Dataset class for QQP"""
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[5], line[3], line[4] # label, text_a, text_b


class load_QNLI_data(data_parser):
    """ Dataset class for QQP"""
    labels = ("entailment", "not_entailment") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[1], line[2] # label, text_a, text_b

class load_RTE_data(data_parser):
    """ Dataset class for RTE"""
    labels = ("entailment", "not_entailment") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[1], line[2] # label, text_a, text_b

class load_WNLI_data(data_parser):
    """ Dataset class for WNLI"""
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[1], line[2] # label, text_a, text_b

def dataset_to_class_mapping(task):
    """ Mapping from task string to Dataset Class """
    table = {'mrpc': load_MRPC_data, 'mnli': load_MNLI_data, 'wnli':load_WNLI_data, 'rte':load_RTE_data, 'qnli':load_QNLI_data, 'qqp':load_QQP_data, 'stsb':load_STSB_data}
    return table[task]

