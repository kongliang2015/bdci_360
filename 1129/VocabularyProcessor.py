from tensorflow.contrib import learn
import numpy as np
max_document_length = 4
x_text =[
    'i love you',
    'me too'
]
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(x_text)
print next(vocab_processor.transform(['i me too'])).tolist()
x = np.array(list(vocab_processor.fit_transform(x_text)))
print x

[1, 4, 5, 0]
[[1 2 3 0]
 [4 5 0 0]]
 
 import numpy as np
from tensorflow.contrib import learn

x_text = ['This is a cat','This must be boy', 'This is a a dog']
max_document_length = max([len(x.split(" ")) for x in x_text])

## Create the vocabularyprocessor object, setting the max lengh of the documents.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

## Transform the documents using the vocabulary.
x = np.array(list(vocab_processor.fit_transform(x_text)))    

## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping

## Sort the vocabulary dictionary on the basis of values(id).
## Both statements perform same task.
#sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])

## Treat the id's as index into list and create a list of words in the ascending order of id's
## word with id i goes at index i of the list.
vocabulary = list(list(zip(*sorted_vocab))[0])

print(vocabulary)
print(x)
