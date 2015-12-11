# summarize.py
# Luke Reichold - CSCI 4930
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import reuters
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk.data
import math
import re

DOC_ROOT = 'docs/'
DEBUG = False
SUMMARY_LENGTH = 5  # number of sentences in final summary
stop_words = stopwords.words('english')
ideal_sent_length = 20.0
stemmer = SnowballStemmer("english")

class Summarizer():

    def __init__(self, articles):

        self._articles = []
        for doc in articles:
            with open(DOC_ROOT + doc) as f:
                headline = f.readline()
                url = f.readline()
                f.readline()
                body = f.read().replace('\n', ' ')
                if not self.valid_input(headline, body):
                    self._articles.append((None, None))
                    continue
                self._articles.append((headline, body))
 

    def valid_input(self, headline, article_text):
        return headline != '' and article_text != ''


    def tokenize_and_stem(self, text):
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered = []

        # filter out numeric tokens, raw punctuation, etc.
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered.append(token)
        stems = [stemmer.stem(t) for t in filtered]
        return stems


    def score(self, article):
        """ Assigns each sentence in the document a score based on the sum of features values.
            Based on 4 features: relevance to headline, length, sentence position, and TF*IDF frequency.
        """

        headline = article[0]
        sentences = self.split_into_sentences(article[1])
        frequency_scores = self.frequency_scores(article[1])

        for i, s in enumerate(sentences):
            headline_score = self.headline_score(headline, s) * 1.5
            length_score = self.length_score(self.split_into_words(s)) * 1.0
            position_score = self.position_score(float(i+1), len(sentences)) * 1.0
            frequency_score = frequency_scores[i] * 4
            score = (headline_score + frequency_score + length_score + position_score) / 4.0
            self._scores[s] = score


    def generate_summaries(self):
        """ If article is shorter than the desired summary, just return the original articles."""

        # Rare edge case (when total num sentences across all articles is smaller than desired summary length)
        total_num_sentences = 0
        for article in self._articles:
            total_num_sentences += len(self.split_into_sentences(article[1]))

        if total_num_sentences <= SUMMARY_LENGTH:
            return [x[1] for x in self._articles]

        self.build_TFIDF_model()  # only needs to be done once

        self._scores = Counter()
        for article in self._articles:
            self.score(article)

        highest_scoring = self._scores.most_common(SUMMARY_LENGTH)
        if DEBUG:
            print(highest_scoring)

        print("## Headlines: ")
        for article in self._articles:
            print("- " + article[0])

        # Appends highest scoring "representative" sentences, returns as a single summary paragraph.
        return ' '.join([sent[0] for sent in highest_scoring])


    ## ----- STRING PROCESSING HELPER FUNCTIONS -----

    def split_into_words(self, text):
        """ Split a sentence string into an array of words """
        try:
            text = re.sub(r'[^\w ]', '', text) # remove non-words
            return [w.strip('.').lower() for w in text.split()]
        except TypeError:
            return None

    def remove_smart_quotes(self, text):
        """ Only concerned about smart double quotes right now. """
        return text.replace(u"\u201c","").replace(u"\u201d", "")


    def split_into_sentences(self, text):
        tok = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tok.tokenize(self.remove_smart_quotes(text))
        sentences = [sent.replace('\n', '') for sent in sentences if len(sent) > 10]
        return sentences


    ## ----- CALCULATING WEIGHTS FOR EACH FEATURE -----

    def headline_score(self, headline, sentence):
        """ Gives sentence a score between (0,1) based on percentage of words common to the headline. """
        title_stems = [stemmer.stem(w) for w in headline if w not in stop_words]
        sentence_stems = [stemmer.stem(w) for w in sentence if w not in stop_words]
        count = 0.0
        for word in sentence_stems:
            if word in title_stems:
                count += 1.0
        score = count / len(title_stems)
        return score


    def length_score(self, sentence):
        """ Gives sentence score between (0,1) based on how close sentence's length is to the ideal length."""
        len_diff = math.fabs(ideal_sent_length - len(sentence))
        return len_diff / ideal_sent_length


    def position_score(self, i, size):
        """ Yields a value between (0,1), corresponding to sentence's position in the article.
            Assuming that sentences at the very beginning and ends of the article have a higher weight. 
            Values borrowed from https://github.com/xiaoxu193/PyTeaser
        """

        relative_position = i / size
        if 0 < relative_position <= 0.1:
            return 0.17
        elif 0.1 < relative_position <= 0.2:
            return 0.23
        elif 0.2 < relative_position <= 0.3:
            return 0.14
        elif 0.3 < relative_position <= 0.4:
            return 0.08
        elif 0.4 < relative_position <= 0.5:
            return 0.05
        elif 0.5 < relative_position <= 0.6:
            return 0.04
        elif 0.6 < relative_position <= 0.7:
            return 0.06
        elif 0.7 < relative_position <= 0.8:
            return 0.04
        elif 0.8 < relative_position <= 0.9:
            return 0.04
        elif 0.9 < relative_position <= 1.0:
            return 0.15
        else:
            return 0


    def build_TFIDF_model(self):
        """ Build term-document matrix containing TF-IDF score for each word in each document
            in the Reuters corpus (via NLTK).
        """
        token_dict = {}
        for article in reuters.fileids():
            token_dict[article] = reuters.raw(article)

        # Use TF-IDF to determine frequency of each word in our article, relative to the
        # word frequency distributions in corpus of 11k Reuters news articles.
        self._tfidf = TfidfVectorizer(tokenizer=self.tokenize_and_stem, stop_words='english', decode_error='ignore')
        tdm = self._tfidf.fit_transform(token_dict.values())  # Term-document matrix


    def frequency_scores(self, article_text):
        """ Individual (stemmed) word weights are then calculated for each
            word in the given article. Sentences are scored as the sum of their TF-IDF word frequencies.
        """

        # Add our document into the model so we can retrieve scores
        response = self._tfidf.transform([article_text])
        feature_names = self._tfidf.get_feature_names() # these are just stemmed words

        word_prob = {}  # TF-IDF individual word probabilities
        for col in response.nonzero()[1]:
            word_prob[feature_names[col]] = response[0, col]
        if DEBUG:
            print(word_prob)

        sent_scores = []
        for sentence in self.split_into_sentences(article_text):
            score = 0
            sent_tokens = self.tokenize_and_stem(sentence)
            for token in (t for t in sent_tokens if t in word_prob):
                score += word_prob[token]

            # Normalize score by length of sentence, since we later factor in sentence length as a feature
            sent_scores.append(score / len(sent_tokens))

        return sent_scores
