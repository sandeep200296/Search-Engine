import index_builder
import re


class QueryEngine:
    def __init__(self, file_names):
        self.file_names = file_names
        self.index = index_builder.IndexBuilder(self.file_names)
        self.inverted_index = self.index.inverted_index
        self.regular_index = self.index.total_index

    def one_word_query(self, word):
        pattern = re.compile('[\W_]+')
        word = pattern.sub(' ', word)
        if word in self.inverted_index.keys():
            return self.rank_results([file_name for file_name in self.inverted_index[word].keys()], word)
        else:
            return []

    def free_text_query(self, string):
        pattern = re.compile('[\W_]+')
        string = pattern.sub(' ', string)
        result = []
        for word in string.split():
            result += self.one_word_query(word)
        return self.rank_results(list(set(result)), string)

    def phrase_query(self, string):
        pattern = re.compile('[\W_]+')
        string = pattern.sub(' ', string)
        listOfLists = []
        result = []
        for word in string.split():
            listOfLists.append(self.one_word_query(word))
        setted = set(listOfLists[0]).intersection(*listOfLists)
        for file_name in setted:
            temp = []
            for word in string.split():
                temp.append(self.inverted_index[word][file_name][:])
            for i in range(len(temp)):
                for idx in range(len(temp[i])):
                    temp[i][idx] -= i
            if set(temp[0]).intersection(*temp):
                result.append(file_name)
        return self.rank_results(result, string)

    def make_vectors(self, documents):
        vectors = {}
        for document in documents:
            document_vector = [0] * len(self.index.get_unique_words())
            for idx, word in enumerate(self.index.get_unique_words()):
                document_vector[idx] = self.index.get_score(word, document)
            vectors[document] = document_vector
        return vectors

    def query_frequency(self, match_word, query):
        cnt = 0
        for word in query.split():
            cnt += (word == match_word)
        return cnt

    def term_frequency(self, terms, query):
        temp = [0] * len(terms)
        for i, term in enumerate(terms):
            temp[i] = self.query_frequency(term, query)
        return temp

    def dot_product(self, doc1, doc2):
        if len(doc1) != len(doc2):
            return 0
        return sum([x * y for x, y in zip(doc1, doc2)])

    def query_vector(self, query):
        pattern = re.compile('[\W_]+')
        query = pattern.sub(' ', query)
        query_list = query.split()
        query_vector = [0] * len(query_list)
        index = 0
        for index, word in enumerate(query_list):
            query_vector[index] = self.query_frequency(word, query)
            index += 1
        query_idf = [self.index.idf[word] for word in self.index.get_unique_words()]
        magnitude = pow(sum(map(lambda x : (x * x), query_vector)), 0.5)
        frequency = self.term_frequency(self.index.get_unique_words(), query)
        tf = [word / magnitude for word in frequency]
        final_vector = [tf[i] * query_idf[i] for i in range(len(self.index.get_unique_words()))]
        return final_vector

    def rank_results(self, result_documents, query):
        vectors = self.make_vectors(result_documents)
        query_vector = self.make_query_vector(query)
        results = [[self.dot_product(vectors[result], query_vector), result] for result in result_documents]
        results.sort(key=lambda x: x[0])
        results = [x[1] for x in results]
        return results


