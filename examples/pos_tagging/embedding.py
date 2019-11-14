import torch
import fasttext
import numpy as np
import fasttext
class EmbeddingReader:
    def __init__(self, embedding_dim, file_path, vocab):
        """
        file_path: Path to embedding file
        vocab: Dictionary mapping word to index
        """
        self.file_path = file_path
        self.word2idx = vocab.word2idx
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.zeros((len(self.word2idx), embedding_dim))

    def get_embedding_matrix(self):
        return self.get_fasttext_tensors()

    def get_fasttext_tensors(self):
        model = fasttext.load_model(self.file_path)
        for word, idx in self.word2idx.items():
            try:
                self.embedding_matrix[idx, :] = model[word]
            except:
                self.embedding_matrix[idx, :] = np.zeros((1, self.embedding_dim))
        return torch.FloatTensor(self.embedding_matrix)