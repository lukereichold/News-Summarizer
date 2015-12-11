# Multi-document news article summarizer

Final project for CSCI 4930 (Machine Learning).

## Usage

Run `python3 driver.py`

Which passes a list of file paths containing docs into Summarizer:
    
    VW_articles = ["VW/VW-ars.txt", "VW/VW-nyt.txt"]
    magic = Summarizer(VW_articles)
    print(magic.generate_summaries())

Outputs a single paragraph, containing a customizable number of sentences extracted from the documents.

**Runtime**: ~ 45 seconds

**Dependencies**: Requires sklearn and NLTK


## Overview

This program uses an unsupervised machine learning algorithm to extract representative sentences from a series of articles to generate a summary. Unlike generative summarization approaches where new content is created, this program's output summary contains only sentences contained in the source documents. Moreover, these summaries are "generic", in that they aren't customized in response to a specific user or query.

With the goal of choosing informative yet non-redundant sentences, each sentence of each set of articles is given a score, weighed by the following features.


### Weighted features for sentence extraction:
1. Words in common with headline (using stemming)
2. Sentence length (assuming longer sentences are more representative; goal: ~20 words).
3. TF-IDF word frequency (using stemming), using 11k Reuters news articles as background corpus.
4. Relative sentence location in article

Each of these features were weighed differently in computing the final sentence score, and were determined by trial-and-error manual testing.


### Design Notes:
- Using NLTK (https://github.com/nltk/nltk) for tokenization and stop-word corpus
- Using scikit-learn for TF-IDF Vectorizer
- Currently using English-language articles only
- Weights for sentence positions borrowed from PyTeaser project (https://github.com/xiaoxu193/PyTeaser/)


### Potential Future Additions:
- While sentence order is a factor in calculating the score of a each sentence in a given article, once the highest ranking sentences from each source are joined, the semantic order is no longer available. The original positions for each sentence could be persisted in the final scores, to produce a final summary whose sentence order reflects that of the initial articles.
- After initially selecting the sentences with highest scores, we might discount TF-IDF scores for duplicate words in the remaining sentences (or in subsequent articles) in effort to reduce repetitiveness in the summary.

