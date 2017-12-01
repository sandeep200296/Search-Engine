'''
    Author : Sandeep Alajangi
    Lang : py
'''

import re
import math

from nltk import PorterStemmer


class IndexBuilder:
    def __init__(self, files):
        self.tf = {}
        self.df = {}
        self.idf = {}
        self.file_names = files
        self.file_to_terms = {}
        self.total_index = self.make_total_index()
        self.inverted_index = self.make_inverted_index()
        self.vectors = self.vectorize()
        self.magnitudes = self.compute_vector_magnitudes(self.file_names)
        self.populate_scores()

    '''
        Returns list of lists
        A list for each file containing the lower_case words after eliminating stop-words and stemming them 
        
        Input : List of file names
        Output : List of list of words, list of words for each file
    '''

    def process_files(self):
        file_to_words = {}

        for file in self.file_names:
            # Regex for valid words.
            pattern = re.compile('[\W_]+')
            file_to_words[file] = open(file, 'r').read().lower()

            # substituting single space instead of the pattern
            file_to_words[file] = pattern.sub(' ', file_to_words[file])
            re.sub(r'[\W_]+', '', file_to_words[file])

            # split the file into list of words
            file_to_words[file] = file_to_words[file].split()

            # Eliminate stopwords
            stop_words = open('stopwords.txt').read().lower()
            temp_words = file_to_words[file]
            file_to_words[file] = [word for word in temp_words if word not in stop_words]

            # Stemming of words
            temp_words = file_to_words[file]
            file_to_words[file] = [PorterStemmer(word) for word in temp_words[file]]

        return file_to_words

    '''
        Returns index of the file
        
        Input : List of words of a file
        Output : Dictionary where keys are the words, and the values are the positions of each word in that file
        
        map < string, vector<int> >
    '''

    def index_file(self, list_of_words):
        file_index = {}
        for index, word in enumerate(list_of_words):
            if word in file_index:
                file_index[word].append(index)
            else:
                file_index[word] = [index]
        return file_index

    '''
        Make indices for all the files 
        Calls the index_file() for all the files
        
        map < file, map <string, vector<int> > >
    '''

    def make_all_indices(self, file_to_words):
        total_index = {}
        for file_name in file_to_words.keys():
            total_index[file_name] = self.index_file(file_to_words[file_name])
        return total_index

    '''
        Creates the Inverted Index
        Calculates tf and idf as well.
    '''

    def make_inverted_index(self):
        inverted_index = {}
        full_index = self.total_index
        for file_name in full_index.keys():
            self.tf = {}
            for word in full_index[file_name].keys():
                self.tf[file_name][word] = len(full_index[file_name][word])
            if word in self.df.keys():
                self.df[word] += 1
            else:
                self.df[word] = 1
            if word in inverted_index.keys():
                if file_name in inverted_index[word].keys():
                    inverted_index[word][file_name].append(full_index[file_name][word][:])
                else:
                    inverted_index[word][file_name] = full_index[file_name][word]
            else:
                inverted_index[word] = {file_name: full_index[file_name][word]}

        return inverted_index

    '''
        Create a vector for each file which keeps count of each word
    '''

    def vectorize(self):
        vectors = {}
        for file_name in self.file_names:
            vectors[file_name] = [len(self.total_index[file_name][word]) for word in self.total_index[file_name].keys()]
        return vectors

    '''
        Returns the number of files in collection
    '''

    def collection_size(self):
        return len(self.file_names)

    '''
        Returns the magnitude of the vector
    '''

    def compute_vector_magnitudes(self, documents):
        magnitudes = {}
        for document in documents:
            magnitudes[document] = pow(sum(map(lambda x: (x * x), self.vectors[document])), 0.5)
        return magnitudes

    def document_frequency(self, word):
        if word in self.inverted_index.keys():
            return len(self.inverted_index[word].keys())
        else:
            return 0

    def term_frequency(self, word, document):
        return self.tf[document][word] / self.magnitudes if word in self.tf[document].keys() else 0

    def compute_idf(self, total_documents, documents_with_terms):
        if documents_with_terms != 0:
            return math.log(total_documents / documents_with_terms)
        else:
            return 0

    def populate_scores(self):
        for file_name in self.file_names:
            for word in self.get_unique_words():
                self.tf[file_name][word] = self.term_frequency(word, file_name)
                if word in self.df.keys():
                    self.idf[word] = self.compute_idf(self.collection_size(), self.df[word])
                else:
                    self.idf[word] = 0
        return self.df, self.tf, self.idf

    def get_score(self, document, word):
        return self.tf[document][word] * self.idf[word]

    def execute(self):
        return self.make_inverted_index()

    def make_total_index(self):
        return self.make_all_indices(self.file_to_terms)

    def get_unique_words(self):
        return self.inverted_index.keys()
