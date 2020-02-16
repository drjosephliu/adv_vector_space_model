import pandas as pd
import scipy.stats as stats
from pymagnitude import *


def main():
    vectors = Magnitude('vectors/GoogleNews-vectors-negative300.magnitude')
    stanf_vectors_50d = Magnitude('vectors/glove.6B.50d.magnitude')
    stanf_vectors_100d = Magnitude('vectors/glove.6B.100d.magnitude')
    stanf_vectors_200d = Magnitude('vectors/glove.6B.200d.magnitude')
    stanf_vectors_300d = Magnitude ('vectors/glove.6B.300d.magnitude')
    stanf_vectors_cc = Magnitude('vectors/glove.840B.300d.magnitude')

    df = pd.read_csv('data/SimLex-999.txt', sep='\t')[['word1', 'word2', 'SimLex999']]
    human_scores = []
    vector_scores = []
    stanf_vector_50d_scores = []
    stanf_vector_100d_scores = []
    stanf_vector_200d_scores = []
    stanf_vector_300d_scores = []
    stanf_vector_cc_scores = []
    for word1, word2, score in df.values.tolist():
        human_scores.append(score)
        similarity_score = vectors.similarity(word1, word2)
        vector_scores.append(similarity_score)

        stanf_50d_sim_score = stanf_vectors_50d.similarity(word1, word2)
        stanf_100d_sim_score = stanf_vectors_100d.similarity(word1, word2)
        stanf_200d_sim_score = stanf_vectors_200d.similarity(word1, word2)
        stanf_300d_sim_score = stanf_vectors_300d.similarity(word1, word2)
        stanf_cc_sim_score = stanf_vectors_cc.similarity(word1, word2)

        stanf_vector_50d_scores.append(stanf_50d_sim_score)
        stanf_vector_100d_scores.append(stanf_100d_sim_score)
        stanf_vector_200d_scores.append(stanf_200d_sim_score)
        stanf_vector_300d_scores.append(stanf_300d_sim_score)
        stanf_vector_cc_scores.append(stanf_cc_sim_score)
        # print(f'{word1},{word2},{score},{similarity_score:.4f}')

    correlation, p_value = stats.kendalltau(human_scores, vector_scores)
    print(f'Correlation = {correlation}, P Value = {p_value}')

    # 1
    least_sim_human = human_scores.index(min(human_scores))
    least_sim_vector = vector_scores.index(min(vector_scores))
    print("===Least similar pair based on human score:===")
    print(df.iloc[least_sim_human])
    print("===Least similar pair based on vector score:===")
    print(df.iloc[least_sim_vector])
    print(vector_scores[least_sim_vector])

    # 2
    most_sim_human = human_scores.index(max(human_scores))
    most_sim_vector = vector_scores.index(max(vector_scores))
    print("===Most similar pair based on human score:===")
    print(df.iloc[most_sim_human])
    print("===Most similar pair based on vector score:===")
    print(df.iloc[most_sim_vector])
    print(vector_scores[most_sim_vector])

    # 3
    stanf_50d_corr, stanf_50d_p_value = stats.kendalltau(human_scores, stanf_vector_50d_scores)
    stanf_100d_corr, stanf_100d_p_value = stats.kendalltau(human_scores, stanf_vector_100d_scores)
    stanf_200d_corr, stanf_200d_p_value = stats.kendalltau(human_scores, stanf_vector_200d_scores)
    stanf_300d_corr, stanf_300d_p_value = stats.kendalltau(human_scores, stanf_vector_300d_scores)
    stanf_cc_corr, stanf_cc_p_value = stats.kendalltau(human_scores, stanf_vector_cc_scores)
    print("===Correlation and p values from different Stanford GloVe embeddings:===")
    print(f'Stanford 50D: correlation = {stanf_50d_corr}, p value = {stanf_50d_p_value}')
    print(f'Stanford 100D: correlation = {stanf_100d_corr}, p value = {stanf_100d_p_value}')
    print(f'Stanford 200D: correlation = {stanf_200d_corr}, p value = {stanf_200d_p_value}')
    print(f'Stanford 300D: correlation = {stanf_300d_corr}, p value = {stanf_300d_p_value}')
    print(f'Stanford Common Crawl: correlation = {stanf_cc_corr}, p value = {stanf_cc_p_value}')


if __name__ == '__main__':
    main()
