"""
Short module that should be run in case you didn't run disambiguate.py
with NER on but still want to use the NER tags. It will use all entries
that begin with he, she or it and replace the it with <NE> and then write
the new dictionary disambiguations_with_ner.yaml.

Note: This will delete the original yaml file.
"""


import yaml

with open("disambiguations.yaml", "r") as stream:
    INP_DICT = yaml.load(stream)


def _convert_to_normalized(dictionary):
    """
    Args:
        - dictionary is a python dict whose values are (integer) numbers.
    Returns:
        - a python dict with the same keys but values which are normalized.
    """
    sum_value = sum(dictionary.values())
    for _key, _value in dictionary.items():
        dictionary[_key] = float("{0:.3f}".format(_value/sum_value))
    return dictionary


OUT_DICT = INP_DICT.copy()
for key, value in INP_DICT.items():
    if (key[0][0] == 'he' or key[0][0] == 'she' or key[0][0] == 'it' or
            key[0][0] == 'they'):
        preposition = key[0][0]
        bias = 1.0
        if preposition == 'it':
            # if the preposition is 'it' let it's values only count half, since
            # most occurences will be names
            # the value is arbitrary
            bias = 0.5
        # this is a bad work-around to get the tuple to actually be a
        # single element as a tuple and not automatically converted to a
        # list.
        new_key = ['placeholder']
        new_key[0] = ("<NE>", key[0][1])
        new_key += key[1:]
        new_key = tuple(new_key)
        value = _convert_to_normalized(value)

        if new_key in OUT_DICT:
            # if the key is already in the dictionary add together the
            # occurence values.
            for key2, value2 in value.items():
                key2 = key2.replace(preposition, '<NE>')
                if key2 in OUT_DICT[new_key]:
                    OUT_DICT[new_key][key2] += bias*value2
                else:
                    OUT_DICT[new_key][key2] = bias*value2
        else:
            # if it's not yet in the dictionary just add it
            OUT_DICT[new_key] = {subkey.replace(preposition, '<NE>'):
                                 subvalue*bias for subkey, subvalue
                                 in value.items()}

# lastly normalize the whole dictionary just for form
for key, value in OUT_DICT.items():
    value = _convert_to_normalized(value)
    OUT_DICT[key] = value

with open("disambiguations.yaml", "w") as stream:
    yaml.dump(OUT_DICT,
              stream,
              explicit_start=True,
              width=79)
