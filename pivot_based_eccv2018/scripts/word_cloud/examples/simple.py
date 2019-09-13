# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Minimal Example
===============

Generating a square wordcloud from the US constitution using default arguments.
"""
import numpy as np
from os import path
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Read the whole text.
#text = open('/home/jxgu/github/unparied_im2text_jxgu/data/mscoco/cocotalk_raw_sents.txt').read()
text = open('/home/jxgu/github/unparied_im2text_jxgu/tmp/20180419-075726.denseatt_en_mscoco_offline.txt').read()
#text = open('/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/nmt_t2t_data_all/train_0303.zh').read().decode('utf8')
#text = open('/home/jxgu/github/unparied_im2text_jxgu/tmp/aic_i2t_zh_caps.txt').read().decode('utf8')

text = text.split()
from collections import Counter
counts = Counter(text)
#alice_coloring = np.array(Image.open("alice_color.png"))
#image_colors = ImageColorGenerator(alice_coloring)
zh_font_path = '/media/jxgu/github/lib/msyh.ttf'
# winter, magma
wordcloud = WordCloud(max_words=100, max_font_size=600,min_font_size=60, font_step=2, width=2000, prefer_horizontal=1.0, height=3000, margin=0, background_color='white', colormap='winter').generate_from_frequencies(counts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
#plt.subplots_adjust(left=0.0, right=0.0, top=0.0, bottom=0.0)
plt.savefig("coco_aic_nmt_en_words.png", dpi=800, bbox_inches='tight')

# The pil way (if you don't have matplotlib)
# image = wordcloud.to_image()
# image.show()
