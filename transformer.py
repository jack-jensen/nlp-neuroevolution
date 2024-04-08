import random
import numpy as np


class Transformer:
    def __init__(self, embeddingDimensionality, queryKeyDimensionality, contextSize):
        self.English = {}
        self.Swedish = {}
        
        self.queryMatrix = np.random.rand(queryKeyDimensionality, embeddingDimensionality)
        self.keyMatrix = np.random.rand(queryKeyDimensionality, embeddingDimensionality)
        self.embeddingDimensionality = embeddingDimensionality
        self.contextSize = contextSize

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
    
    def sinusoidalPE(self, positionOfToken, dimensions):
        return np.sin(positionOfToken/10000 ** (2 * dimensions/dimensions))
        
    
    def inputEmbedding(self):
        for word in self.words:
            embeddingMatrix = np.vstack((embeddingMatrix, self.English[word]))
            
    
    def attentionBlock(self):
        relavanceMatrix = np.zeros(self.contextSize, self.contextSize)
        for y in range(self.contextSize):
            yWordKey = self.embeddingMatrix[self.words[y]] * self.keyMatrix
            for x in range(self.contextSize):
                xWordKey = self.embeddingMatrix[self.words[x]] * self.keyMatrix
                
                
                
    

    
    
        
        
               
            
        
        
transformer = Transformer(5)
transformer.textPreprocessing("The lazy fox jumped over the fat ugle doll")
print(transformer.embeddingMatrix)