#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains the necessary functions to load  a text-corpus from
NLTK, contract all possible sentences, applying POS-tags to the
contracted sentences and compare that with the original text.
The information about which contraction+pos-tag pair gets expanded to
which full form will be saved in a dictionary for use in expander.py
"""

__author__ = "Yannick Couzinié"

# standard library imports
import pprint
import yaml
# third-party library imports
import nltk
# local library imports
import utils

# increase the allowed ram size that the models can use
# nltk.internals.config_java(options='-xmx2G')


def _find_sub_list(sublist, full_list):
    """
    Args:
        - sublist is a list of words that are supposed to be found in
          the full list.
        - full list is a list of words that is supposed to be searched
          in.
    Returns:
        - List of tuples with the form
            (first_index_of_occurence, last_index_of_occurence)

    This function finds all occurences of sublist in the full_list.
    """
    # this is the output list
    results = []
    sublist_len = len(sublist)
    # loop over all ind if the word in full_list[ind] matches the first
    # word of the sublist
    for ind in (i for i, word in enumerate(full_list)
                if word == sublist[0]):
        # check that the complete sublist is matched
        if full_list[ind:ind+sublist_len] == sublist:
            # then append this to the results
            results.append((ind, ind+sublist_len-1))
    return results


def _contract_sentences(expansions,
                        sent_lst,
                        use_ner,
                        ner_args):
    """
    Args:
        - expansions is a dictionary containing  the corresponding
          contractions to the expanded words
        - sent_lst is a list of sentences, which is itself a list of
          words, i.e. [["I", "am", "blue"], [...]].
        - use_ner is boolean to decide whether to use
          named-entity-recognition for a potential increase in
          accuracy but with the obvious costs of performance.
        - ner_args is a list with an  object of StanfordNERTagger and
          the tag to be used. This only needs to be
          supplied if use_ner is true.

    Returns:
        - yields tuples of the form
              (index of first word that was replaced,
              list of words that were replaced,
              contracted sentence).
          The above example would then give
              (0, ["I", "am"], ["I", "'m", "blue"])
          Note that uncontractible sentences are not added to the
          output.
          Since yield is used, iterate over the results. Otherwise it
          takes too much time.

    This function checks a list of sentences for whether they can be
    contracted. It starts with the first two words, then the first three
    and then goes on to the second+third, then the second+third+fourth
    and so on.
    """
    # first find the indices of the sentences that contain contractions

    for sent in sent_lst:
        if use_ner:
            # replace all named entities with the tag in ner_args[1]
            # throw away replacement info
            sent = utils.sent_to_ner(ner_args[0], sent,
                                     tag=ner_args[1])[0]
        # check whether any expansion is present then add the index
        # it has a True for every expansion that is present
        expansion_bool = [expansion in ' '.join(sent) for expansion
                          in list(expansions.keys())]
        if not any(expansion_bool):
            # if no expansions present just continue
            continue

        # convert the boolean list to a list of indices
        expansion_idx = [i for i, boolean in enumerate(expansion_bool)
                         if boolean]

        # the list of relevant expansions for the sentence
        relevant_exp = [list(expansions.keys())[i] for i in expansion_idx]
        for expansion in relevant_exp:
            # first split the contraction up into a list of the same
            # length as the expanded string
            if len(expansion.split()) in [2, 3, 4]:
                # if you contract three or two words,
                # just split at apostrophes
                contraction = expansions[expansion].split("'")
                assert len(contraction) == len(expansion.split())
                # add the apostrophes again
                contraction[1] = "'" + contraction[1]
                if len(contraction) == 3:
                    contraction[2] = "'" + contraction[2]
                if len(contraction) == 4:
                    contraction[3] = "'" + contraction[3]
            else:
                # this case is only entered when there is only one word
                # input. So assert that this is the case.
                assert len(expansion) == 1
                # this is a completely pathological case, since
                # ambiguous 1-word replacements are not in the common
                # list of replacements from wikipedia. But since one can
                # openly expand contractions.yaml it is checked.
                contraction = expansions[expansion]
            # find where the sublist occurs
            occurences = _find_sub_list(expansion.split(), sent)
            # loop over all first indices of occurences
            # and insert the contracted part
            for occurence in occurences:
                contr_sent = sent[:occurence[0]] + contraction
                contr_sent += sent[occurence[0]+len(contraction):]
                yield (occurence[0],
                       sent[occurence[0]:occurence[0]+len(contraction)],
                       contr_sent)


def _invert_contractions_dict():
    """
    This is just a short function to return the inverted dictionary
    of the contraction dictionary.
    """
    with open("contractions.yaml", "r") as stream:
        # load the dictionary containing all the contractions
        contractions = yaml.load(stream)

    # invert the dictionary for quicker finding of contractions
    expansions = dict()
    for key, value in contractions.items():
        if len(value) == 1:
            continue
        for expansion in value:
            if expansion in expansions:
                print("WARNING: As an contraction to {}, {} is replaced with"
                      " {}.".format(expansion,
                                    expansions[expansion],
                                    key))
            expansions[expansion] = key
    return expansions


def write_dictionary(pos_model,
                     sent_lst,
                     add_tags=0,
                     use_ner=False,
                     ner_args=None):
    """
    Args:
        - pos_model is an instance of StanfordPOSTagger
        - sent-lst a list of sentences which themselves are lists of the
          single words.
        - add_tags is the amount of pos tags used after the
          relevant contraction, this can be used to further
          disambiguate but (of course) spreads out the data.
        - use_ner is boolean to decide whether to use
          named-entity-recognition for a potential increase in
          accuracy but with the obvious costs of performance.
        - ner_args is a list with an  object of StanfordNERTagger and
          the tag to be used. This only needs to be
          supplied if use_ner is true.

    Returns:
        - None, but writes a disambiguations.yaml file with disambiguations
          for the ambiguous contractions in contractions.yaml.

    Raises:
        ValueError if use_ner is True but no ner_model is supplied.

    Using the provided list of sentences, contract them and pos-tag them.
    Using the pos-tags it is then possible to classify which
    (contraction, pos-tag) combinations get expanded to which ambiguous
    long form.
    """
    # pylint: disable=too-many-locals
    if use_ner and (ner_args is None):
        raise ValueError("The use_ner flag is True but no NER"
                         " model has been supplied!")

    expansions = _invert_contractions_dict()

    output_dict = dict()
    ambiguity_counter = 0
    for tuple_rslt in _contract_sentences(expansions,
                                          sent_lst,
                                          use_ner=use_ner,
                                          ner_args=ner_args):
        # pos tag the sentence
        if use_ner:
            # first replace the NER tag with "it"
            pos_sent = [word.replace(ner_args[1], "it") for word
                        in tuple_rslt[2]]
            # tag the sentence
            pos_sent = pos_model.tag(pos_sent)
            # and replace it with the tag again
            pos_sent = [(tuple_rslt[2][i], word_pos[1]) for i, word_pos
                        in enumerate(pos_sent)]
        else:
            pos_sent = pos_model.tag(tuple_rslt[2])
        # extract the pos tags on the contracted part
        contr_word_pos = pos_sent[tuple_rslt[0]:(tuple_rslt[0] +
                                                 len(tuple_rslt[1]))]
        if add_tags == 0:
            contr_pos = tuple(contr_word_pos)
        else:
            add_pos_list = pos_sent[len(tuple_rslt[1]):(len(tuple_rslt[1]) +
                                                        add_tags)]
            add_pos = [pos_word[1] for pos_word in add_pos_list]
            contr_pos = tuple(contr_word_pos + add_pos)
        # write a dictionary entry connecting the (words, pos) of the
        # contraction to the expanded part
        word = ' '.join(tuple_rslt[1])
        if contr_pos not in output_dict:
            output_dict[contr_pos] = dict()
            output_dict[contr_pos][word] = 1
            # keep track of the progress
            print("\n\n ---- \n\n")
            pprint.pprint(output_dict)
            print("Ambiguity counter is {}.".format(ambiguity_counter))
            print("\n\n ---- \n\n")
        elif word in output_dict[contr_pos].keys():
            # check whether the entry is already there
            output_dict[contr_pos][word] += 1
            continue
        else:
            # if the combination of pos tags with words already occured
            # once then a list has to be made. Ideally this case doesn't
            # occur
            ambiguity_counter += 1
            output_dict[contr_pos][word] = 1
            print("\n\n ---- \n\n")
            print("AMBIGUITY ADDED!")
            pprint.pprint(output_dict)
            print("Ambiguity counter is {}.".format(ambiguity_counter))
            print("\n\n ---- \n\n")
    with open("disambiguations.yaml", "w") as stream:
        yaml.dump(output_dict, stream)


if __name__ == '__main__':
    # if you call this function directly just build the disambiguation
    # dictionary.
    # load a corpus that has the form of list of sentences which is
    # split up into a list of words
    SENT_LST = nltk.corpus.brown.sents()
    SENT_LST += nltk.corpus.gutenberg.sents()
    SENT_LST += nltk.corpus.reuters.sents()
    SENT_LST += nltk.corpus.inaugural.sents()
    POS_MODEL = utils.load_stanford('pos')
    NER_MODEL = utils.load_stanford('ner')
    write_dictionary(POS_MODEL,
                     SENT_LST,
                     add_tags=1,
                     use_ner=False,
                     ner_args=[NER_MODEL, "<NE>"])
