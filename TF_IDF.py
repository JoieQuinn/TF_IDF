#!/usr/bin/env python3

'''
Creates a .json containing token frequency counts, inverse document
frequency counts, and a tf*idf score for a corpus of textfiles.
'''

from __future__ import print_function
import re
from json import dump
from os import listdir
from collections import Counter
from os.path import join, realpath, dirname
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

this_directory = dirname(realpath(__file__))


def freq_dist(wordlist):
    """
    Generate a frequency dist on wordlist.

    :param wordlist: list of words
    :type wordlist: list

    :returns: frequency distribution dictionary mapping n-gram strings
              to frequencies
    :rtype: Counter
    """

    return Counter(wordlist)


def reduce_dicts(dict1, master_dict):
    """
    Sum values for keys in dict1 and master_dict

    :param dict1: dictionary of token frequencies from individual file.
    :type dict1: dictionary
    :param master_dict: cumulative dictionary of token frequencies
                for all files in corpus
    :type master_dict: dictionary

    :returns: master distrbution mapping individual tokens to
            frequencies over entire corpus.
    :rtype: Counter
    """

    return sum((Counter(y) for y in [dict1, master_dict]), Counter())


def strip_punctuation(text):
    """
    Remove punctuation from txt file.

    :param text: contents of txt file, joined with ' '
    :type token: string

    :returns: list of tokens with punctuation removed
    :rtype: list
    """

    text = re.sub(r"[^\w\'\-]+", " ", text)
    return text.split()


def tf_idf(dict1, dict2):
    """
    Combine a dictionary of term frequencies with a dictionary of
    inverse document frequencies. Calculate the tf*idf of each word.

    :param dict1: dictionary of term frequencies
    :type dict1: dictionary
    :param dict2: dictionary of inverse document frequencies
    :type dict2: dictionary

    :returns: a dictionary of values for term frequency, inverse
        document frequncy, and tf*idf for each key in the tf_idf
        dictionary
    :rtype: dictionary
    """

    tf_idf = {}
    for term in dict1:
        tf = dict1[term]
        idf = dict2[term]
        tf_idf[term] = {'tf': tf, 'idf': idf, 'tfidf': tf*idf}
    return tf_idf


def stem_words(text):
    """
    Stem words in file using the Porter Stemmer.

    :param text: contents of txt file, joined with ' '
    :type token: string

    :returns: a list of stemmed words
    :rtype: list
    """

    ps = PorterStemmer()
    tokens = word_tokenize(text)
    words = [ps.stem(word) for word in tokens]
    return words


def remove_stopwords(text, stopwords):
    """
    Compares a list of words with a predefined list of "stop words"
    and removes the stop words from the list.

    :param text: list of words in a file
    :type token: list
    :param stopwords: list of stopwords in English
    :param stopwords: list

    :returns: list words with stopwords removed.
    :rtype: list
    """

    return [word for word in text if word not in stopwords]


def main():
    parser = ArgumentParser(conflict_handler='resolve',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Generates .json file containing '
                                        'token frequences (TF), inverse '
                                        'document frequencies (IDF), and '
                                        'the TF*IDF score for each token in '
                                        'each txt file in a directory.')
    parser.add_argument('-i', '--input_dir',
                        required=True,
                        help='Identifies input directory of files. Required',
                        type=str)
    parser.add_argument('-p', '--remove_punct',
                        help='Removes punctuation before processing each '
                             'file, leaving the original txt file unchanged.',
                        action='store_true',
                        default=False)
    parser.add_argument('-s', '--stem',
                        help='Uses the Porter Stemmer to return counts for '
                             'root words, leaving the original txt files '
                             'unchanged.  This option automatically removes '
                             'punctuation.',
                        action='store_true',
                        default=False)
    parser.add_argument('-l', '--remove_stopwords',
                        help='Removes common words from each file before '
                             'processing, leaving the original txt files '
                             'unchanged',
                        action='store_true',
                        default=False)
    parser.add_argument('-a', '--all_options',
                        help='Removes stopwords and punctuation and stems '
                             'words before processing, leaving the original '
                             'txt files unchanged',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    Remove_Stemmed_Stops = False

    #Consolidate user options 
    if args.all_options:
        args.remove_punct = True
        args.stem = True
        args.remove_stopwords = True
        Remove_Stemmed_Stops = True

    args.remove_punct = (args.remove_punct
                        or args.stem
                        or args.remove_stopwords)

    #If user forces all options from command line
    if args.remove_stopwords and args.stem:
        Remove_Stemmed_Stops = True

    input_dir = realpath(args.input_dir)

    tf_dict_all = {}
    dfs_dict = {}
    stops = set(stopwords.words("english"))
    if Remove_Stemmed_Stops:
        stops_string = ' '.join(stops)
        stemmed_stops = set(stem_words(stops_string))
        stemmed_stops = stops.union(stemmed_stops)

    for file_name in listdir(input_dir):
        file_path = join(input_dir, file_name)

        with open(file_path) as f:
            words = f.read().lower().strip().split()            
            words = " ".join(words)

            if args.remove_punct:
                words = strip_punctuation(words)
            if args.remove_stopwords:
                if Remove_Stemmed_Stops:
                    words = remove_stopwords(words, stemmed_stops)
                else:
                    words = remove_stopwords(words, stops)
            if args.stem:
                words = stem_words(' '.join(words))

            freq_counts = freq_dist(words)

            # calculate cumulative frequency over all documents
            tf_dict_all = reduce_dicts(freq_counts, tf_dict_all)

            # create document frequency counts
            for word in set(words):
                dfs_dict[word] = dfs_dict.get(word, 0) + 1

            # calculate inverse document frequency counts
            for item in dfs_dict:
                doc_frequency = dfs_dict[item]
                inverse_doc_frequency = 1/doc_frequency
                dfs_dict[item] = inverse_doc_frequency

    # Combine dictionaries.
    tf_idfs = tf_idf(dict(tf_dict_all), dict(dfs_dict))

    # Write dictionary of tf, idf, tf*idf scores to .json file called
    # 'output.json' in the same directory as the script.
    with open("output.json", 'w') as f:
        dump(tf_idfs, f)

if __name__ == '__main__':
    main()
