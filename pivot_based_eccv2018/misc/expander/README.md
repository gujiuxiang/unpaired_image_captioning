# Expander
## A small module to expand common contractions in the english language

This is the expander module with it's main feature the function
`expand_contractions` in `expand.py`. It uses an object of the
`StanfordPOSTagger` class from `nltk` to POS-tag input sentences and
decide accordingly which expansion to use.

To be able to run the code you need to download a Stanford POS-tagger
model. You can download the basic english tagger on the [official
homepage](https://nlp.stanford.edu/software/tagger.shtml "Stanford POS
                Tag official website"). 
Furthermore to enhance the POS-tagging the named-entity recognition
model from the Stanford Core NLP set is used as well, it can be
downloaded on it's respective [official 
homepage](https://nlp.stanford.edu/software/CRF-NER.shtml "Stanford NER
                official website").
Extract the zip-file(s) into the subdirectory `stanford_models` of this module.
Alternatively, you can supply the path to the model in the call to
`load_stanford` as documented in the program.

To see example output run `expand.py` directly using `python expand.py`.
You can supply your own directory to the call of `load_stanford()`
here. In this you can also see how to use this module.

### Assumptions being made

- Apostrophes in the middle of a lexical item (i.e. *usually*
  sequences of characters surrounded by spaces and/or delimited 
  by punctuation) are signs for contraction and will be dealt
  with as such. 
- The input sentence is grammatically correct.
- The only replacements needed to be done are defined in
  `contractions.yaml`
  
### Notable drawbacks

- The nature of using POS-taggers is of course, that they are
  not perfect. The best is being done to make correct
  expansions, but errors will happen. Especially since
  expansions are not unambiguous.

### TODOs

- Include a test case when `expander.py` is run directly, 
  correctly asserting that the right results come out.
- Write a function that divides list at certain characters (apostrophe
                in our case), and refactor code with it.
- Combine the he-she-it cases to one central `<HSI>` case in order to
  get more test cases and thusly improve accuracy (may not be sensible, as the
  cases for he/she and it are different).
- Adapt `load_stanford` in `utils.py` to use the new CoreNLPPOSTagger and
  CoreNLPNERTagger instead of the deprecated ones.

### Notes about Licensing

This software is distributed unter the Apache 2.0 license, mainly because NLTK
is as well and it seems to allow enough freedom. 
**Note though that the stanford
models are not distributed under that license.** They are full GPL and restrict
any kind of proprietary use. If you intend to use this software in your own
proprietary software, either get in contact with the people at stanford or
rewrite the program to use models included in NLTK (if you are doing that, I
would also be grateful for a pull request with the changes).
I have just generally found the stanford models to be more reliable.
