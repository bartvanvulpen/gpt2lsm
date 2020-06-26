import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from more_itertools import split_before
import numpy as np
import string
import time
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                          GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)


class LSMLoss:

    def __init__(self, input_ids, output_logits, token_type_ids, labels, tokenizer):
        """
        Initializing LSM Loss
        :param input_ids: input ids of the input batch
        :param output_logits: output logits of the batch
        :param token_type_ids: token type ids
        :param: labels: the label responses in the dataset
        :param tokenizer: (GPT2) tokenizer object
        """
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.output_logits = output_logits
        self.labels = labels
        self.tokenizer = tokenizer
        self.batch_size = input_ids.size()[0]


    def get_fw_dict(self):
        """
        Get dictionary of function words
        :return: Dictionary of function words
         """
        d = {
            'conjunctions': ['and', 'but', 'or', 'as', 'if', 'when', 'because', 'while', 'where', 'although', 'whether',
                             'before', 'since', 'so', 'though', 'until', 'after', 'cos', 'for', '&', 'nor', 'unless',
                             'once', 'whereas', 'whilst', 'rather than', 'and/or', 'even when', 'albeit', 'given that',
                             'provided that'],
            'auxiliary_verbs': ['be', 'is', 'are', 'were', 'was', 'been', 'am', 'being', 'have', 'has', 'was', 'were',
                                'would', 'will', 'do', 'can', 'could', 'dare', 'does', 'did', 'had', 'having', 'may',
                                'might', 'must', 'need', 'ought', 'shall', 'should', "'ll", "'d"],
            'personal_pronouns': ['i', 'you', 'he', 'they', 'she', 'we', 'who', 'them', 'him', 'me', 'her', 'us',
                                  'himself', 'themselves', 'someone', 'herself', 'anyone', 'everyone', 'whom', 'myself',
                                  'each other', 'yourself', 'no one', 'somebody', 'nobody', 'everybody', 'anybody',
                                  'his', 'mine', 'ourselves', 'yours', 'one another', 'hers', 'no-one', 'ours',
                                  'theirs', 'his', 'their', 'her', 'my', 'your', 'our'],
            'impersonal_pronouns': ['it', 'its', 'they', 'that', 'this', 'them', 'something', 'nothing', 'anything',
                                    'itself', 'themselves', 'itself', 'everything', 'each other', 'everything',
                                    'something', "'em"],
            'prepositions': ['of', 'in', 'to', 'for', 'with', 'on', 'by', 'at', 'from', 'as', 'into', 'about', 'like',
                             'after', 'between', 'through', 'over', 'against', 'under', 'without', 'within', 'during',
                             'before', 'towards', 'around', 'upon', 'including', 'among', 'across', 'off', 'behind',
                             'since', 'rather than', 'until', 'according to', 'up to', 'despite', 'near', 'above',
                             'per', 'along', 'away from', 'throughout', 'outside', 'round', 'beyond', 'worth', 'down',
                             'on to', 'up', 'due to', 'inside', 'plus'],
            'adverbs': ['so', 'up', 'then', 'out', 'now', 'only', 'just', 'more', 'also', 'very', 'well', 'how', 'down',
                        'back', 'on', 'there', 'still', 'even', 'too', 'here', 'where', 'however', 'over', 'in', 'as',
                        'most', 'again', 'never', 'why', 'off', 'really', 'always', 'about', 'when', 'quite', 'much',
                        'both', 'often', 'away', 'perhaps', 'right', 'already', 'yet', 'later', 'almost', 'of course',
                        'far', 'together', 'probably', 'today', 'actually', 'ever', 'at least', 'enough', 'less',
                        'for example', 'therefore', 'particularly', 'either', 'around', 'rather', 'else', 'sometimes',
                        'thus', 'ago', 'yesterday', 'home', 'all', 'usually'],
            'quantifiers': ['all', 'some', 'any', 'many', 'more', 'another', 'much', 'each', 'few', 'most', 'both',
                            'several', 'half', 'little', 'whatever', 'less', 'enough', 'either', 'fewer', 'neither',
                            'a lot', 'least', 'a bit', 'a great deal', 'plenty'], 'articles': ['a', 'an', 'the']}

        return d

    def get_relevant_parts(self):
        """
        This function gets the relevant parts of the input, token_type_ids, output and labels,
        that contain actual words.
        :return: relevant for each of the aforementioned parts
        """
        batch_size = self.input_ids.size()[0]
        inp, tti, outp, lbls = [], [], [], []

        for i in range(batch_size):
            inp.append(self.input_ids[i][-1])
            tti.append(self.token_type_ids[i][-1])
            lbls.append(self.labels[i][-1])
            outp.append(self.output_logits[i][-1])

        return torch.stack(inp), torch.stack(tti), torch.stack(outp), torch.stack(lbls)


    def encode_fw_dict(self, fw_d):
        """
        This function encodes the function word dictionary with the GPT2 tokenizer.
        Adds <bos> for correct encoding, then later removes it.
        :param fw_d: dictionary of function words for each category
        :return: encoded dictionary tensor + its length
        """
        flatten = lambda l: [item for sublist in l for item in sublist]

        encoded_d = []
        for key, value in fw_d.items():

            # encode dictionary value
            fws1 = ' ' + ' '.join(value)
            fw_ids1 = self.tokenizer.encode(fws1)

            # also encode in another way, without weird G before the word
            fw_ids2 = flatten([self.tokenizer.encode(x) for x in value])

            # combine two lists and remove duplicates
            fw_ids = torch.tensor(list(set(fw_ids1 + fw_ids2))).cuda()

            # scatter 1.0 at indices in vocabulary to create distribution
            fw_distr_sm = F.softmax(torch.cuda.FloatTensor(len(self.tokenizer)).fill_(0).scatter(0, fw_ids, 1.0), dim=-1)
            encoded_d.append(fw_distr_sm)

        return torch.stack(encoded_d), len(encoded_d)

    def get_output_idx_focus(self):
        """
        This function gets, using the labels, the indices of where
        the relevant output words are generated in the output.
        """
        idxs = []
        for x in self.labels:
            idxs.append((x != -100).nonzero().view(-1))
        return idxs

    def get_sp2_hist_focus(self):
        """
        This function gets the output indices that belong to speaker 2 (the conversation partner)
        :return: the output indices belonging to speaker 2
        """
        sp2, pad = self.tokenizer.convert_tokens_to_ids(['<speaker2>', '<pad>'])
        idxs = []
        for i, x in enumerate(self.token_type_ids):
            sp2tag_idcs = (self.input_ids[i] == sp2).nonzero().view(-1)
            sp2_hist = (x == sp2).nonzero().view(-1)
            sp2_hist = torch.stack([idc for idc in sp2_hist if idc not in sp2tag_idcs])
            idxs.append(sp2_hist)

        return idxs


    def get_sp1_history(self):
        """
        This function gets the history of speaker 1 (chatbot) from the relevant input_ids, by padding
        the history of speaker2.
        :return: history of speaker 1
        """
        inp = self.input_ids
        sp1, sp2, pad = self.tokenizer.convert_tokens_to_ids(['<speaker1>', '<speaker2>', '<pad>'])
        flatten = lambda l: [item for sublist in l for item in sublist]

        # get history of speaker1 (so pad all history of speaker2)
        hist1 = torch.zeros(inp.size(), device='cuda',dtype=torch.long)

        # for each training example in batch, get sp1 history
        for index, example in enumerate(inp):
            splitted = list(split_before(example.cpu().numpy(), lambda x: x == sp1 or x == sp2))
            for i, z in enumerate(splitted):
                # replace sp1 tag with pad tag
                z = [x if x != sp1 else pad for x in z]
                splitted[i] = z
                # if sp2 or last sp1, replace with pad
                if sp2 in z or i == len(splitted) - 1:
                    splitted[i] = [pad] * len(z)

            hist1[index] = torch.tensor(flatten(splitted), dtype=torch.long)

        # also return sp1_ids in last sentence of each batch
        idxs = []
        for example in hist1:
            idxs.append((example != pad).nonzero().view(-1))

        return hist1.view(inp.size()), idxs


    def mask_sp1tags_in_input(self):
        """
        This functions masks the <speaker1> tags in the relevant input ids, so when comparing the input
        (<speaker2> utterances/ history) with the output (<speaker1> history (actually in input too) + next utterance),
        the focus will be on <speaker2> only.
        :return: masked relevant input ids
        """
        sp1, sp2, pad = self.tokenizer.convert_tokens_to_ids(['<speaker1>', '<speaker2>', '<pad>'])
        x = self.input_ids.view(-1)
        t = self.token_type_ids.view(-1)
        indices_sp1 = (t == sp1).nonzero().view(-1)
        indices_sp2tag = (x == sp2).nonzero().view(-1)
        # also replace <speaker2> tags with <pad>
        masked = x.scatter(0, indices_sp1, pad).scatter(0, indices_sp2tag, pad)


        return masked.view(self.input_ids.size())

    def encode_ids(self, ids):
        """
        This function encodes ids into a one hot vector of the vocabulary.
        Then to create a softmax'ed distribution, without zeros (to avoid division by zero errors).
        :param ids: input_ids
        :return: Softmax of encoded input id's
        """
        enc_input_ids = torch.cuda.FloatTensor(len(ids), len(self.tokenizer)).fill_(0).scatter(1, ids.unsqueeze(-1),
                                                                                               1.0)
        return F.softmax(enc_input_ids, dim=-1)

    def reshape_input_output(self):
        """
        This function reshapes the tensor with input ids, then encodes it with encode_ids, that
        return the log-softmax of the input distribution, which will be used for the KL divergence measure.
        It also reshapes the tensor with output logits and returns its log-softmax
        :return: reshaped input and output (log-softmax)
        """
        # shape input
        input_ids_reshaped = self.input_ids.view(-1)
        input = self.encode_ids(input_ids_reshaped)

        # shape output
        lm_logits_reshaped = self.output_logits.view(-1, self.output_logits.size(-1))
        output = F.softmax(lm_logits_reshaped, dim=-1)  # to be used for KL divergence

        return input, output


    def calc_kl_divs(self, d, distr, o=False):
        """
        This function calculates for each distribution of a word in the batch of sentences the
        inverted KL Divergence value, in order to give a measure of how likely the distribution
        of this word is to be from a specific function word category.
        :param d: dictionary with function words
        :param distr: distributions of the words in a batch of sentences (distr_size = size of vocab)
        :return: inverted kl divergence value for each word in the sentence/batch of sentences, for each category
        """
        distr = distr.detach()

        # avoid log zero error by adding a little bit of value to zero values (occurs in output_distr only)
        if o:
            distr[distr == 0] = 1e-40

        # calculate KL Divergence values for given distribution
        measures_output = []
        for fw in d:
            fw_r = fw.repeat(distr.size()[0], 1)
            m = (fw_r * (fw_r / distr).log()).sum(1)
            measures_output.append(m.pow(-1))
        kl_divs = torch.stack(measures_output)

        return kl_divs.view(len(d), self.batch_size, -1)

    def loss(self):
        """
        This function combines all functions above and calculates the LSM loss.
        :return: LSM loss and kl_div values (to trace during training)
        """

        # get load and encode dict of function words
        d = self.get_fw_dict()
        d, d_len = self.encode_fw_dict(d)

        # get relevant parts from the inputs
        self.input_ids, self.token_type_ids, self.output_logits, self.labels = self.get_relevant_parts()

        # get focus of output logits + focus on sp2 history during LSM calculation
        focus_output_sp1 = self.get_output_idx_focus()
        focus_hist_sp2 = self.get_sp2_hist_focus()

        # get history distributions of speaker1
        sp1_hist, focus_hist_sp1 = self.get_sp1_history()
        sp1_hist = self.encode_ids(sp1_hist.view(-1))

        # mask input_ids, reshape input_ids tensor, also reshape output logits tensor
        self.input_ids = self.mask_sp1tags_in_input()
        input, output = self.reshape_input_output()

        # calc inverted KL divs
        kl_divs_input = self.calc_kl_divs(d, input, o=False) # padded speaker1 utterances, so speaker2
        kl_divs_output = self.calc_kl_divs(d, output, o=True) # new/next utterance
        kl_divs_chatbot_hist = self.calc_kl_divs(d, sp1_hist, o=True) # speaker1 history only

        # calculate LSM losses for each sentence and each category
        losses = []
        for i, cat in enumerate(kl_divs_input):
            sp1 = 0
            sp2 = 0
            for j, example in enumerate(cat):
                last_hist_sp2 = example
                last_output = kl_divs_output[i][j]
                last_hist_sp1 = kl_divs_chatbot_hist[i][j]

                last_hist_sp1 = torch.stack([last_hist_sp1[z] for z in focus_hist_sp1[j]])
                last_hist_sp2 = torch.stack([last_hist_sp2[z] for z in focus_hist_sp2[j]])
                last_output = torch.stack([last_output[z] for z in focus_output_sp1[j]])

                sp1_total = torch.cuda.FloatTensor(np.concatenate((last_hist_sp1.cpu().numpy(), last_output.cpu().numpy()), 0))

                sp1 += torch.mean(sp1_total)
                sp2 += torch.mean(last_hist_sp2)

            cat_LSM = 100 * (abs(sp1 - sp2) / (sp1 + sp2))

            losses.append(cat_LSM)

        # final loss is the mean of all categories
        final_lsm_loss = torch.mean(torch.stack(losses))

        return final_lsm_loss, kl_divs_input[0][0], kl_divs_output[0][0]