import json
import collections
import random

import numpy as np
import torch.utils.data.dataset as Dataset
import torch
import torch.nn as nn


class Vocabulary(object):
    def __init__(self, vocab_file, num_relations, num_entities):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.num_relations = num_relations
        self.num_entities = num_entities
        assert len(self.vocab) == self.num_relations + self.num_entities + 2

    def load_vocab(self, vocab_file):
        """
        Load a vocabulary file into a dictionary.
        """
        vocab = collections.OrderedDict()
        fin = open(vocab_file, encoding='utf-8')
        for num, line in enumerate(fin):
            items = line.strip().split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def convert_by_vocab(self, vocab, items):
        """
        Convert a sequence of [tokens|ids] using the vocab.
        """
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)

    def convert_token_to_id(self, token):
        return self.vocab[token]

    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)

    def __len__(self):
        return len(self.vocab)


class NaryExample(object):
    def __init__(self,
                 arity,
                 head,
                 relation,
                 tail,
                 auxiliary_info=None):
        self.arity = arity
        self.head = head
        self.relation = relation
        self.tail = tail
        self.auxiliary_info = auxiliary_info


class NaryFeature(object):
    def __init__(self,
                 feature_id,
                 example_id,
                 input_tokens,
                 input_ids,
                 input_mask,
                 mask_position,
                 mask_label,
                 mask_type,
                 arity,
                 max_node_num,
                 incidence_matrix,
                 fact_neighbor_tokens_list

                 ):
        """
        Construct NaryFeature.
        Args:
            feature_id: unique feature id
            example_id: corresponding example id
            input_tokens: input sequence of tokens
            input_ids: input sequence of ids
            input_mask: input sequence mask
            mask_position: position of masked token
            mask_label: label of masked token
            mask_type: type of masked token,
                1 for entities (values) and -1 for relations (attributes)
            arity: arity of the corresponding example
        """
        self.feature_id = feature_id
        self.example_id = example_id
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.mask_position = mask_position
        self.mask_label = mask_label
        self.mask_type = mask_type
        self.arity = arity
        self.max_node_num = max_node_num
        self.incidence_matrix = incidence_matrix
        self.fact_neighbor_tokens_list = fact_neighbor_tokens_list


def read_examples(input_file, max_arity):
    """
    Read a n-ary json file into a list of NaryExample.
    """
    examples, total_instance = [], 0
    with open(input_file, "r", encoding='utf-8') as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            arity = obj["N"]
            relation = obj["relation"]
            head = obj["subject"]
            tail = obj["object"]

            auxiliary_info = None
            if arity > 2:
                auxiliary_info = collections.OrderedDict()

                for attribute in sorted(obj.keys()):
                    if attribute in ("N", "relation", "subject", "object"):
                        continue

                    auxiliary_info[attribute] = sorted(obj[attribute])
            if arity <= max_arity:
                example = NaryExample(
                    arity=arity,
                    head=head,
                    relation=relation,
                    tail=tail,
                    auxiliary_info=auxiliary_info)
                examples.append(example)

                total_instance += (2 * (arity - 2) + 3)
    return examples, total_instance


def convert_examples_to_features(examples, vocabulary, max_arity, max_seq_length, sample_k, max_node_num):
    """
    Convert a set of NaryExample into a set of NaryFeature. Each single
    NaryExample is converted into (2*(n-2)+3) NaryFeature, where n is
    the arity of the given example.
    """
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, \
        "Each input sequence contains relation, head, tail, " \
        "and max_aux attribute-value pairs."

    features = []
    feature_id = 0
    
    fact_vab_id_dict, token_fact_id = get_token_fact_id(examples, vocabulary)

    for (example_id, example) in enumerate(examples):

        hrt = [example.head, example.relation, example.tail]
        hrt_mask = [1, 1, 1]


        h_neighbor_set = set()
        r_neighbor_set = set()
        t_neighbor_set = set()
        for i, token_hrt in enumerate(hrt):
            if i == 0:
                h_neighbor_set.update(set(token_fact_id[vocabulary.convert_token_to_id(token_hrt)]))
            elif i == 1:
                r = set(token_fact_id[vocabulary.convert_token_to_id(token_hrt)])
                h = set(token_fact_id[vocabulary.convert_token_to_id(hrt[0])])
                t = set(token_fact_id[vocabulary.convert_token_to_id(hrt[2])])
                r_neighbor_set.update((r & h) | (r & t))
            elif i == 2:
                t_neighbor_set.update(set(token_fact_id[vocabulary.convert_token_to_id(token_hrt)]))

        aux_q = []
        aux_q_mask = []
        aux_values = []
        aux_values_mask = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_q.append(attribute)
                    aux_q.append(value)
                    aux_q_mask.append(1)
                    aux_q_mask.append(1)

        while len(aux_q) < max_aux * 2:
            aux_q.append("[PAD]")
            aux_q.append("[PAD]")
            aux_q_mask.append(0)
            aux_q_mask.append(0)
        assert len(aux_q) == max_aux * 2

        orig_input_tokens = hrt + aux_q
        orig_input_mask = hrt_mask + aux_q_mask
        

        assert len(orig_input_tokens) == max_seq_length and len(orig_input_mask) == max_seq_length

        sample_k = sample_k

        for mask_position in range(max_seq_length):
            if orig_input_tokens[mask_position] == "[PAD]":
                continue

            mask_label = vocabulary.vocab[orig_input_tokens[mask_position]]

            mask_type = 1 if mask_position % 2 == 0 else -1

            input_tokens = orig_input_tokens[:]
            input_tokens[mask_position] = "[MASK]"

            h_neighbor_list = get_neighbor_list(h_neighbor_set, sample_k, example_id)
            r_neighbor_list = get_neighbor_list(r_neighbor_set, sample_k, example_id)
            t_neighbor_list = get_neighbor_list(t_neighbor_set, sample_k, example_id)
            h_neighbor_set_k = set(h_neighbor_list)
            r_neighbor_set_k = set(r_neighbor_list)
            t_neighbor_set_k = set(t_neighbor_list)
            all_neighbor_facts_tokens_dict = collections.defaultdict(list)
            if mask_position == 0:
                h_neighbor_facts_tokens_dict = get_facts_tokens_id_mask(h_neighbor_set_k, fact_vab_id_dict,
                                                                        mask_position)

                neighbor_sets = list(r_neighbor_set_k | t_neighbor_set_k)
                neighbor_facts_tokens_dict = get_facts_tokens_id(neighbor_sets, fact_vab_id_dict)

                h_neighbor_facts_tokens_dict_values = list(h_neighbor_facts_tokens_dict.values())
                neighbor_facts_tokens_dict_values = list(neighbor_facts_tokens_dict.values())
                for i, h_neighbor_facts_tokens_dict_item in enumerate(h_neighbor_facts_tokens_dict_values):
                    all_neighbor_facts_tokens_dict[i] = h_neighbor_facts_tokens_dict_item
                for i, neighbor_facts_tokens_dict_item in enumerate(neighbor_facts_tokens_dict_values):
                    all_neighbor_facts_tokens_dict[
                        i + len(h_neighbor_facts_tokens_dict_values)] = neighbor_facts_tokens_dict_item

            elif mask_position == 1:
                r_neighbor_facts_tokens_dict = get_facts_tokens_id_mask(r_neighbor_set_k, fact_vab_id_dict,
                                                                        mask_position)

                neighbor_sets = list(h_neighbor_set_k | t_neighbor_set_k)
                neighbor_facts_tokens_dict = get_facts_tokens_id(neighbor_sets, fact_vab_id_dict)

                r_neighbor_facts_tokens_dict_values = list(r_neighbor_facts_tokens_dict.values())
                neighbor_facts_tokens_dict_values = list(neighbor_facts_tokens_dict.values())
                for i, r_neighbor_facts_tokens_dict_item in enumerate(r_neighbor_facts_tokens_dict_values):
                    all_neighbor_facts_tokens_dict[i] = r_neighbor_facts_tokens_dict_item
                for i, neighbor_facts_tokens_dict_item in enumerate(neighbor_facts_tokens_dict_values):
                    all_neighbor_facts_tokens_dict[
                        i + len(r_neighbor_facts_tokens_dict_values)] = neighbor_facts_tokens_dict_item
            elif mask_position == 2:
                t_neighbor_facts_tokens_dict = get_facts_tokens_id_mask(t_neighbor_set_k, fact_vab_id_dict,
                                                                        mask_position)

                neighbor_sets = list(h_neighbor_set_k | r_neighbor_set_k)
                neighbor_facts_tokens_dict = get_facts_tokens_id(neighbor_sets, fact_vab_id_dict)

                t_neighbor_facts_tokens_dict_values = list(t_neighbor_facts_tokens_dict.values())
                neighbor_facts_tokens_dict_values = list(neighbor_facts_tokens_dict.values())
                for i, t_neighbor_facts_tokens_dict_item in enumerate(t_neighbor_facts_tokens_dict_values):
                    all_neighbor_facts_tokens_dict[i] = t_neighbor_facts_tokens_dict_item
                for i, neighbor_facts_tokens_dict_item in enumerate(neighbor_facts_tokens_dict_values):
                    all_neighbor_facts_tokens_dict[
                        i + len(t_neighbor_facts_tokens_dict_values)] = neighbor_facts_tokens_dict_item
            else:
                neighbor_sets = list(h_neighbor_set_k | r_neighbor_set_k | t_neighbor_set_k)

                all_neighbor_facts_tokens_dict = get_facts_tokens_id(neighbor_sets, fact_vab_id_dict)

            max_node_num, incidence_matrix, fact_neighbor_tokens_list = get_incidence_matrix(
                all_neighbor_facts_tokens_dict,
                max_node_num,
                input_tokens,
                vocabulary, sample_k)

            input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
            assert len(input_tokens) == max_seq_length and len(input_ids) == max_seq_length

            feature = NaryFeature(
                feature_id=feature_id,
                example_id=example_id,
                input_tokens=input_tokens,
                input_ids=input_ids,
                input_mask=orig_input_mask,
                mask_position=mask_position,
                mask_label=mask_label,
                mask_type=mask_type,
                arity=example.arity,
                max_node_num=max_node_num,
                incidence_matrix=incidence_matrix,
                fact_neighbor_tokens_list=fact_neighbor_tokens_list
            )
            features.append(feature)
            feature_id += 1

    return features


class MultiDataset(Dataset.Dataset):
    def __init__(self, vocabulary: Vocabulary, examples, max_arity=2, max_seq_length=3, sample_k=10,max_node_num=95):
        self.examples = examples
        self.vocabulary = vocabulary
        self.max_arity = max_arity
        self.max_seq_length = max_seq_length
        self.sample_k = sample_k
        self.max_node_num = max_node_num

        self.features = convert_examples_to_features(
            examples=self.examples,
            vocabulary=self.vocabulary,
            max_arity=self.max_arity,
            max_seq_length=self.max_seq_length,
            sample_k=self.sample_k,
        max_node_num = self.max_node_num)
        self.multidataset = []
        for feature in self.features:
            feature_out = [feature.input_ids] + [feature.input_mask] + \
                          [feature.mask_position] + [feature.mask_label] + [feature.mask_type] + [
                              feature.max_node_num] + [feature.incidence_matrix] + [feature.fact_neighbor_tokens_list]
            self.multidataset.append(feature_out)

    def __len__(self):
        return len(self.multidataset)

    def __getitem__(self, index):
        x = self.multidataset[index]

        batch_data = prepare_batch_data(x, self.vocabulary, self.max_arity, self.max_seq_length)
        return batch_data





def prepare_batch_data(inst, vocabulary: Vocabulary, max_arity, max_seq_length):
    input_ids = np.array(inst[0]).astype("int64")
    input_mask = np.array(inst[1]).astype("int64")
    mask_position = np.array(inst[2]).astype("int64")
    mask_label = np.array(inst[3]).astype("int64")
    query_type = np.array(inst[4]).astype("int64")
    input_mask = np.outer(input_mask, input_mask).astype("bool")
    max_node_num = np.array(inst[5]).astype("int64")
    incidence_matrix = np.array(inst[6]).astype("int64")
    fact_neighbor_tokens_list = np.array(inst[7]).astype("int64")

    edge_labels = []
    max_aux = max_arity - 2
    edge_labels.append([0, 1, 2] + [3, 4] * max_aux)
    edge_labels.append([1, 0, 5] + [6, 7] * max_aux)
    edge_labels.append([2, 5, 0] + [8, 9] * max_aux)


    for idx in range(max_aux):
        edge_labels.append([3, 6, 8] + [11, 12] * idx + [0, 10] + [11, 12] * (max_aux - idx - 1))
        edge_labels.append([4, 7, 9] + [12, 13] * idx + [10, 0] + [12, 13] * (max_aux - idx - 1))
    edge_labels = np.asarray(edge_labels).astype("int64")

    mask_output = np.zeros(len(vocabulary.vocab)).astype("bool")
    if query_type == -1:
        mask_output[2:2 + vocabulary.num_relations] = True
    else:
        mask_output[2 + vocabulary.num_relations:] = True

    return input_ids, input_mask, mask_position, mask_label, mask_output, edge_labels, query_type, max_node_num, incidence_matrix, fact_neighbor_tokens_list


def get_token_fact_id(examples, vocabulary):
    fact_vab_id_dict = collections.defaultdict(list)
    token_fact_id = collections.defaultdict(list)
    for examples_index, example in enumerate(examples):
        orig_input_token, _ = get_orig_input_token(example)
        tokens_ids = vocabulary.convert_tokens_to_ids(orig_input_token)
        fact_vab_id_dict[examples_index] = tokens_ids
        for tokens_id in tokens_ids:
            token_fact_id[tokens_id].append(examples_index)
    return fact_vab_id_dict, token_fact_id


def get_orig_input_token(example):
    hrt = [example.head, example.relation, example.tail]
    orig_token_type = [0, 1, 2]
    aux_q = []
    if example.auxiliary_info is not None:
        for attribute in example.auxiliary_info.keys():
            for value in example.auxiliary_info[attribute]:
                aux_q.append(attribute)
                orig_token_type.append(3)
                aux_q.append(value)
                orig_token_type.append(4)

    orig_input_token = hrt + aux_q
    return orig_input_token, orig_token_type


def get_neighbor_list(h_neighbor_set, sample_k, example_id):
    neighbor_list = list(h_neighbor_set - {example_id}) if len(h_neighbor_set) > 0 else h_neighbor_set
    if sample_k < len(neighbor_list):
        sample_neighbor_list = random.sample(neighbor_list, sample_k)
    else:
        sample_neighbor_list = neighbor_list
    return sample_neighbor_list


def get_facts_tokens_id(neighbor_list, fact_vab_id_dict):
    fact_tokens = collections.defaultdict(list)
    for fact_idx, token in enumerate(neighbor_list):
        fact_tokens[fact_idx] = fact_vab_id_dict[token]
    return fact_tokens


def get_facts_tokens_id_mask(h_neighbor_set_k, fact_vab_id_dict, mask_position):
    fact_tokens = collections.defaultdict(list)
    for fact_idx, token_id in enumerate(h_neighbor_set_k):
        each_neighbor_tokens = fact_vab_id_dict[token_id].copy()
        each_neighbor_tokens[mask_position] = 1
        fact_tokens[fact_idx] = each_neighbor_tokens
    return fact_tokens


def get_incidence_matrix(neighbor_facts_tokens_dict, max_node_num, input_tokens,
                         vocabulary, sample_k):


    fact_tokens_list = vocabulary.convert_tokens_to_ids(input_tokens)
    fact_neighbor_tokens_list = []
    fact_neighbor_tokens_list.extend(fact_tokens_list)
    for fact_tokens in neighbor_facts_tokens_dict.values():
        add_tokens = set(fact_tokens) - set(fact_neighbor_tokens_list)
        fact_neighbor_tokens_list.extend(add_tokens)

    if len(fact_neighbor_tokens_list) > max_node_num:
        max_node_num = len(fact_neighbor_tokens_list)


    fact_tokens_list_index = [fact_neighbor_tokens_list.index(id) for id in fact_tokens_list if id != 0]
    hyperedge = torch.zeros(max_node_num)
    hyperedge[fact_tokens_list_index] = 1
    incidence_matrix_list = []
    incidence_matrix_list.append(hyperedge)

    for nei_index, nodes in neighbor_facts_tokens_dict.items():
        nodes_index = [fact_neighbor_tokens_list.index(id) for id in nodes]
        hyperedge = torch.zeros(max_node_num)
        hyperedge[nodes_index] = 1
        incidence_matrix_list.append(hyperedge)
    while len(incidence_matrix_list) < sample_k * 3 + 1:
        incidence_matrix_list.append(torch.zeros(max_node_num))

    while len(fact_neighbor_tokens_list) < max_node_num:
        fact_neighbor_tokens_list.append(0)

    incidence_matrix = torch.stack(incidence_matrix_list).t()

    return max_node_num, incidence_matrix, fact_neighbor_tokens_list


def batch_to_block_diagonal_sparse(tensor):
    """
    Convert a tensor of shape [batch_size, N, M] to a block diagonal matrix
    and then convert it to a torch.sparse_coo_tensor.

    Args:
    - tensor (torch.Tensor): Input tensor of shape [batch_size, N, M]

    Returns:
    - torch.sparse_coo_tensor: Output sparse tensor of shape [batch_size*N, batch_size*M]
    """
    batch_size, N, M = tensor.shape
    total_N = batch_size * N
    total_M = batch_size * M

    indices = []
    values = []

    for b in range(batch_size):

        batch_indices = torch.nonzero(tensor[b], as_tuple=False)
        batch_indices[:, 0] += b * N
        batch_indices[:, 1] += b * M
        indices.append(batch_indices)

        values.append(tensor[b][tensor[b] != 0])

    indices_node_edge = torch.cat(indices, dim=0).t()
    values = torch.cat(values, dim=0)
    indices_edge_node = indices_node_edge[[1, 0], :]
    sparse_tensor = torch.sparse_coo_tensor(indices_node_edge, values, size=(total_N, total_M))
    return sparse_tensor, indices_node_edge, indices_edge_node
