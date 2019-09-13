#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" General utilities file """

__author__ = "Yannick Couzinié"


# standard library imports
import glob
import os
# third party library imports
import nltk


def load_stanford(model,
                  model_name=None,
                  dirname=None):
    """
    Args:
        model - either "pos" or "ner" for pos-tagging or named entity
                recognition.
        model_name - Name of the model to be used, if none is supplied,
                     the recommended standard is loaded.
        dirname - Directory name where the model is located, if none is
                  supplied it is assumed to be in ./stanford_models.
    Returns:
        An object of Stanford(POS/NER)Tagger
    Raises:
        LookupError if the model is not found.

    Load the Stanford module specified by model.
    For this you have to download the model from:

        https://nlp.stanford.edu/software/stanford-postagger-2017-06-09.zip
        https://nlp.stanford.edu/software/stanford-ner-2017-06-09.zip

    respectively and unzip the containers into the stanford_model
    sub-directory of this model or alternatively in the specified
    dirname.
    """
    if model == 'pos':
        sub_dir = "models"
        jar_name = "stanford-postagger.jar"
        if model_name is None:
            model_name = "english-bidirectional-distsim.tagger"
    elif model == 'ner':
        sub_dir = "classifiers"
        jar_name = "stanford-ner.jar"
        # the model name can be adapted to use the 4class or 7class
        # model for recognition. The 3 class model has been deemed the
        # most stable one for now.
        if model_name is None:
            model_name = "english.all.3class.distsim.crf.ser.gz"
    else:
        raise ValueError("Illegal model name in call to load_stanford(), "
                         "use 'pos' or 'ner'.")

    if dirname is None:
        # set model dir to ./asset/models
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(current_dir, "stanford_models")
    else:
        model_dir = dirname
    # glob for the class recursively in model_dir
    glob_dir = os.path.join(model_dir,
                            "**",
                            jar_name)
    classes = glob.glob(glob_dir,
                        recursive=True)

    if len(classes) > 1:
        # only have one stanford postagger version at any time
        raise LookupError("Multiple {} versions found, please only have one"
                          " version downloaded at any time.".format(model))
    # point model_dir to the models directory in the stanford pos
    # directory
    model_dir = os.path.dirname(classes[0])
    model_dir = os.path.join(model_dir, sub_dir)

    if not os.path.exists(model_dir):
        raise LookupError("The model directory could not be found.")
    # set the environment variables so that nltk can find the models
    os.environ["CLASSPATH"] = classes[0]
    os.environ["STANFORD_MODELS"] = model_dir

    # load the model
    if model == 'pos':
        stanford_model = nltk.tag.StanfordPOSTagger(model_name)
    elif model == 'ner':
        stanford_model = nltk.tag.StanfordNERTagger(model_name)
    return stanford_model


def conv_2_word_pos(stanford_model,
                    sent_list,
                    is_split,
                    use_ner,
                    ner_args=None):
    """
    Args:
        stanford_model - object of StanfordPOSTagger, as returned by
                         load_stanford_pos.
        sent_list - List with sentences split up into list of singular words
                    e.g. [["I", "am", "sentence", "one"],
                          ["I", "am", "sentence", "two"]]
        is_split - if False then the input should be
                         ["I am sentence one",
                          "I am sentence two"]
        use_ner - boolean to decide whether to use
                  named-entity-recognition for a potential increase in
                  accuracy but with the obvious costs of performance.
        ner_args - is a list with an  object of StanfordNERTagger and
                   the tag to be used. This only needs to be
                   supplied if use_ner is true.

    Returns:
           output - the same list of sentences, with each word replaced by
                     a (word, tag) tuple.

    Raises:
        ValueError if use_ner is True but no ner_args is supplied.

    Converts a sentence list to a list of lists, where the lists contain
    (word, pos_tag) tuples using the provided stanford_model.
    """
    if use_ner and ner_args is None:
        raise ValueError("The use_ner flag is True but no NER"
                         " model has been supplied!")
    for sent in sent_list:
        if not is_split:
            tmp = nltk.word_tokenize(sent)
            if use_ner:
                tmp_ner = sent_to_ner(ner_args[0], tmp, tag=ner_args[1])
                # work with the sentence which has the named entity tag
                sent = tmp_ner[0]

            for i, word in enumerate(tmp):
                j = 0
                # This while loop breaks once there is no apostrophe
                # left anymore.
                while "'" in word[1:-1] and word != "n't":
                    # search for ' in the middle of the word indicating
                    # that the splitting is not correct. An example is:
                    #   Who'd've -> Who'd, 've
                    # so we need to check and split up further.
                    tmp2 = tmp[:i+j]
                    # if this is not the first apostrophe, take care to
                    # add the apostrophe again
                    if j:
                        tmp2.append("'" + word.split("'", 1)[0])
                    else:
                        tmp2.append(word.split("'", 1)[0])
                    tmp2.append("'" + word.split("'", 1)[1])
                    word = word.split("'", 1)[1]
                    tmp2 += tmp[i+j+1:]
                    tmp = tmp2
                    j += 1
                sent = tmp
        # if one uses the NER return the information about what was replaced
        # as well.
        if use_ner:
            # using ner it is usually better to have known words in the
            # POS-tagger so replace the tag with 'it'. This is what will
            # increase accuracy of the pos tags and is the main reason for
            # using NER.
            tmp = [e.replace(ner_args[1], "it") for e in sent]
            tmp = stanford_model.tag(tmp)
            # now put the pos tags on the output with the <NE> tag rather
            # than "it".
            tmp = [(word, tmp[i][1]) for i, word in enumerate(sent)]
            # and finally make this a nested tuple with the replacement
            # information attached
            yield (tmp, tmp_ner[1])
        else:
            yield stanford_model.tag(tmp)


def sent_to_ner(ner_model, sent, tag="<NE>"):
    """
    Args:
        - ner_model is an object of StanfordNERTagger as usually
          returned by load_stanford('ner').
        - sent is the sentence, broken up into a list of its words,
          that shall have it's named entities replaced with tag.
        - tag is the tag with which to replace the entities. Usually
          this will be <NE>.
    Returns:
        - sent with tag instead of the named entities.

    This function takes an input sentence analyzes it with the NER
    tagger and replaces any word that is not tokenized with 'O' (the
    trivial object) with <NE>. Examples for that are in the 3class
    classifier with 'PERSON', 'LOCATION', 'ORGANIZATION'.
    """
    # first let the tagger run
    ner_tagged = ner_model.tag(sent)
    # extract all indices of words that are not tagged with 'O'
    idx_lst = [i for i, word_ner in enumerate(ner_tagged)
               if word_ner[1] != 'O']
    output = sent
    replaced = []
    for idx in idx_lst:
        # insert the tag
        output[idx] = tag
        replaced.append(ner_tagged[idx][0])
    return (output, replaced)


def ner_to_sent(sent, replaced, tag="<NE>"):
    """
    Args:
        - sent is the sentence that has the NER tags in them instead of
          the actual named entities.
        - replaced is the corresponding list of named entities that
          will be inserted in the order of appearance for the tags.
        - tag is the tag that was used to replace the named entities.

    Returns:
        - sent but with it's original named entities

    Raises:
        - ValueError if the amount of <NE> and the length of the
          replacement list do not match.

    This function serves to convert the result from sent_to_ner back to
    a sentence with the named entities in it.
    """
    if len(replaced) != sent.count(tag):
        raise ValueError("The wrong replaced list has been provided"
                         " to ner_to_sent.")

    for i, word in enumerate(sent):
        if word == tag:
            # pop makes sure that every replacement is inserted exactly
            # once
            sent[i] = replaced.pop(0)
    return sent
