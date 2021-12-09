import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, RandomizedSearchCV,KFold
seed = 123
import json
#settings
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=np.inf)
pd.options.display.max_columns = None
pd.options.display.max_rows = None

#Defining the pretrained distil bert model
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


##Loading the json file:
with open('/home/ubuntu/TERM_PROJECT/intents.json') as json_file:
    data = json.load(json_file)
    print(data)
data = data['intents']


#Reading the json file and converting it into the dataframe:
dataset = pd.DataFrame(columns=['tag', 'patterns', 'responses'])
for i in data:
    tag = i['tag']
    for t, r in zip(i['patterns'], i['responses']):
        row = {'tag': tag, 'patterns': t, 'responses':r}
        dataset = dataset.append(row, ignore_index=True)
print(dataset['tag'].head(250))

tokenized = dataset['patterns'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
np.array(padded).shape
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape
#create an input tensor out of the padded token matrix, and send that to DistilBERT
input_ids = torch.tensor(np.array(padded))
with torch.no_grad():
  # last_hidden_states holds the outputs of DistilBERT.
  # It is a tuple with the shape (number of examples, max number of tokens in the sequence, number of hidden units in the DistilBERT model).
    last_hidden_states = model(input_ids)
 # Slice the output for the first position for all the sequences, take all hidden unit outputs
features = last_hidden_states[0][:,0,:].numpy()

# Split our datset into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(features, dataset['tag'], test_size=0.2, random_state=42)


# search for the best value of the C parameter, which determines regularization strength.
parameters = {'C': np.linspace(0.0001, 100, 20)}

lr_finder = RandomizedSearchCV(estimator = LogisticRegression() , scoring = 'accuracy',
                               param_distributions = parameters,
                               cv = KFold(n_splits=5, shuffle=True, random_state = seed),
                               verbose=50, random_state=seed, n_jobs = -1)
lr_finder.fit(X_train, y_train)

print('best parameters: ', lr_finder.best_params_)
print('best scores: ', lr_finder.best_score_)


lr_cv = lr_finder.best_estimator_.fit(X_train, y_train)
y_lr_pred = lr_cv.predict(X_test)
print(metrics.classification_report(y_test, y_lr_pred))