# Quick Notes on this file:
# 1. This file contains the components that may be called on to create the final structure
# 2. It should have been "Quick Note" without the 's'


import random
import numpy as np


class Transformer:
    def __init__(self, embeddingDimensionality, queryKeyDimensionality, contextSize):
        self.English = {}
        self.Swedish = {}
        self.contextSize = contextSize
        
        self.embeddingMatrix = None
        self.queryMatrix = np.random.rand(queryKeyDimensionality, embeddingDimensionality)
        self.keyMatrix = np.random.rand(queryKeyDimensionality, embeddingDimensionality)
        self.valueDownMatrix = np.random.rand(embeddingDimensionality, queryKeyDimensionality)
        self.valueUpMatrix = np.random.rand(queryKeyDimensionality, embeddingDimensionality)
        
        self.embeddingDimensionality = embeddingDimensionality
        self.positionalMatrix = np.zeros((contextSize, embeddingDimensionality))
        self.positionalMatrixGenerator()
        

    #region PREPROCESSING ENGLISH
    def textPreprocessing(self, string):
        self.sentence = string.lower()
        self.words = self.sentence.split()
        for word in self.words:
            if not word in self.English:
                self.English[word] = self.randomSemanticVector(self.embeddingDimensionality)
                
    def randomSemanticVector(self, dimensions):
        vector = []
        for i in range(0, dimensions):
            vector.insert(i, random.uniform(-1, 1))
            
        vector = np.array(vector)
            
        return vector
    #endregion
    
    def positionalMatrixGenerator(self):
        for pos in range(self.contextSize):
            for i in range(self.embeddingDimensionality):
                if i % 2 == 0:
                    self.positionalMatrix[pos, i] = np.sin(pos / (10000 ** (2 * i / self.embeddingDimensionality)))
                else:
                    self.positionalMatrix[pos, i] = np.cos(pos / (10000 ** (2 * i / self.embeddingDimensionality)))
                    
        
        
    
    def inputEmbeddingEnglish(self):
        for word in self.words:
            if self.embeddingMatrix is None:
                self.embeddingMatrix = self.English[word]
            else:
                self.embeddingMatrix = np.vstack((self.embeddingMatrix, self.English[word]))
        return self.embeddingMatrix
    
    

        
    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.
        
        Parameters:
        x -- A numpy array of shape (n_samples, n_features)
        
        Returns:
        softmax_probs -- A numpy array of shape (n_samples, n_features) 
                        where each row contains softmax probabilities corresponding to input scores.
        """
        # Numerically stable softmax
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax_probs = e_x / np.sum(e_x, axis=1, keepdims=True)
        
        return softmax_probs
        
            
    
    def multiHeadedSelfAttentionBlock(self, numberOfHeads, queryMatrixDict, keyMatrixDict, valueMaticesDict, embeddingMatrixOrigin, embeddingMatrixEnd):
        #TODO: This only grabs the attention for a single language -- keys need to be from the swedish dictionary
        #The matrices are dictionarys where the key is the head in reference and the value is a numpy array
        
        #Idea -- Make this cross attention by simply entering the valueMatricesDict from the other language?
        
        for i in range(numberOfHeads):
            
            queryMatrix = queryMatrixDict[i]
            keyMatrix = keyMatrixDict[i]
            valueDownMatrix = valueMaticesDict[i][0]
            valueUpMatrix = valueMaticesDict[i][1]
            
            relavanceMatrix = np.zeros(self.contextSize, self.contextSize)
            for y in range(self.contextSize):
                WordKey = embeddingMatrix[y] * self.keyMatrix
                for x in range(self.contextSize):
                    WordQuery = embeddingMatrix[x] * self.queryMatrix
                    
                    relavanceMatrix[y, x] = np.dot(WordKey, WordQuery)
            
            relavanceMatrix = self.softmax(relavanceMatrix)
            
            
            for y in range(self.contextSize):
                WordValue = embeddingMatrix[y] * self.valueDownMatrix * self.valueUpMatrix
                relavanceMatrix[y, :] *= WordValue
                
            for x in range(self.contextSize):
                embeddingMatrix[x] += relavanceMatrix[:, x]
                
                
    # def multiHeadedCrossAttentionBlock(self, numberOfHeads, queryMatrix)
            
            
            
        
            
        
        
        
        
                
        
                
                
        
transformer = Transformer(5, 3, 10)
transformer.textPreprocessing("The lazy fox jumped over the fat ugle doll")
print(transformer.positionalMatrix)