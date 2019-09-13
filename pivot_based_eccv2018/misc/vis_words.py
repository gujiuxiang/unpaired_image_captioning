import scattertext as ST
import pandas as pd
import io
import tarfile, urllib, io
from IPython.display import IFrame
import nltk
import spacy
from gensim.models import word2vec
from scattertext import SampleCorpora, word_similarity_explorer_gensim, Word2VecFromParsedCorpus
from scattertext.CorpusFromParsedDocuments import CorpusFromParsedDocuments

def graph():
  f = open("data/mscoco/output_cocotalk_sents.txt", "r")
  inputfile = f.read()
  tokens = nltk.tokenize.word_tokenize(inputfile)
  fd = nltk.FreqDist(tokens)
  fd.plot(30,cumulative=False)

def vis():
    '''
    text1 = open("/home/jxgu/github/unparied_im2text_jxgu/tmp/aic_nmt_val_5k_zh.en.txt", "r").read()
    text2 = open("/home/jxgu/github/unparied_im2text_jxgu/tmp/aic_nmt_val_5k_zh_online.en.txt", "r").read()
    df = pd.DataFrame( [{'text': text.strip(), 'label': 'text1'} for text in text1.decode('utf-8', errors='ignore').split('\n')] + [{'text': text.strip(), 'label': 'text2'} for text in text2.decode('utf-8', errors='ignore').split('\n')]
    )
    term_doc_mat = ST.TermDocMatrixFromPandas(data_frame = df, category_col = 'label', text_col = 'text', nlp = ST.whitespace_nlp ).build()
    filtered_term_doc_mat = (ST.TermDocMatrixFilter(pmi_threshold_coef = 1, minimum_term_freq = 1).filter(term_doc_mat))
    scatter_chart_data = (ST.ScatterChart(filtered_term_doc_mat).to_dict('text1', category_name='text1', not_category_name='text2'))
    viz_data_adapter = ST.viz.VizDataAdapter(scatter_chart_data)
    html = ST.viz.HTMLVisualizationAssembly(viz_data_adapter).to_html()
    open('subj_obj_scatter.html', 'wb').write(html.encode('utf-8'))
    IFrame(src='subj_obj_scatter.html', width = 1000, height=1000)
    '''

    SUBJECTIVITY_URL = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz'
    data = io.BytesIO(urllib.urlopen(SUBJECTIVITY_URL).read())
    tarball = tarfile.open(fileobj=data, mode = 'r:gz')
    readme = tarball.extractfile('subjdata.README.1.0').read()
    quote = tarball.extractfile('quote.tok.gt9.5000').read()
    plot = tarball.extractfile('plot.tok.gt9.5000').read()

    text1 = open("tmp/flickr_test_1k_zh.en.txt", "r").read()
    text2 = open("tmp/flickr_test_1k_zh.en.txt", "r").read()
    # Examples of subjective sentences in corpus
    #quote.decode('utf-8', errors='ignore').split('\n')[:3]
    '''Construct subjective vs. objective pandas dataframe, 
    treating review quotes as subjective, and plot points as objective.
    '''
    df = pd.DataFrame(
        [{'text': text.strip(), 'label': 'subjective'} for text
         in quote.decode('utf-8', errors='ignore').split('\n')]
        + [{'text': text.strip(), 'label': 'objective'} for text
           in plot.decode('utf-8', errors='ignore').split('\n')]
    )
    '''Convert Pandas dataframe to a term-document matrix, indicating
    the category column is "label" and the text column name is "text".'''
    nlp = spacy.load('en')

    corpus = ST.CorpusFromPandas(data_frame = df,
                                              category_col = 'label',
                                              text_col = 'text',
                                              # Note: use nlp=spacy.en.English() for text that's not pre-tokenized
                                              nlp=nlp
                                              ).build()
    term_freq_df = corpus.get_term_freq_df()

    html = ST.produce_scattertext_explorer(corpus,
              category='label',
              category_name='subjective',
              not_category_name='objective',
              width_in_pixels=1000)
    open("Convention-Visualization.html", 'wb').write(html.encode('utf-8'))

def gensim_similarity():
    nlp = spacy.load('en')
    convention_df = SampleCorpora.ConventionData2012.get_data()
    convention_df['parsed'] = convention_df.text.apply(nlp)
    corpus = CorpusFromParsedDocuments(convention_df, category_col='party', parsed_col='parsed').build()
    model = word2vec.Word2Vec(size=300,
                              alpha=0.025,
                              window=5,
                              min_count=50,
                              max_vocab_size=None,
                              sample=0,
                              seed=1,
                              workers=1,
                              min_alpha=0.0001,
                              sg=1,
                              hs=1,
                              negative=0,
                              cbow_mean=0,
                              iter=1,
                              null_word=0,
                              trim_rule=None,
                              sorted_vocab=1)
    html = word_similarity_explorer_gensim(corpus,
                                           category='democrat',
                                           category_name='Democratic',
                                           not_category_name='Republican',
                                           target_term='jobs',
                                           minimum_term_frequency=50,
                                           pmi_threshold_coefficient=4,
                                           width_in_pixels=1000,
                                           metadata=convention_df['speaker'],
                                           word2vec=Word2VecFromParsedCorpus(corpus, model).train(),
                                           max_p_val=0.05,
                                           save_svg_button=True)
    open('./demo_gensim_similarity.html', 'wb').write(html.encode('utf-8'))