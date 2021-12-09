import os
import math
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
from sklearn.metrics import confusion_matrix, classification_report
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

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

##Building the classes:

from sklearn.model_selection import train_test_split

#dataset_train, dataset_validate, dataset_test = \
              #np.split(dataset.sample(frac=1, random_state=42),
                       #[int(.6*len(dataset)), int(.8*len(dataset))])
#dataset_train , dataset_test = train_test_split(dataset , test_size=0.1)
#print(dataset_train)

##building train_csv seperately:
df = pd.DataFrame(dataset)
df.to_csv('train_data.csv', index=False,header=('tag','patterns','responses'))
train=pd.read_csv('train_data.csv')
print(len(train))
print(train.head())


##building the test_csv seperately:
df = pd.DataFrame(dataset)
df.to_csv('test_data.csv', index=False,header=('tag','patterns','responses'))
test=pd.read_csv('test_data.csv')
print(len(test))
print(test.head())


#building the valid_csv seperately:
#df = pd.DataFrame(dataset_validate)
#df.to_csv('valid_data.csv', index = False, header=('tag','patterns','responses'))
#valid = pd.read_csv('valid_data.csv')
#print(len(valid))
#print(valid.head())

##creating a model directory:
os.makedirs("/home/ubuntu/TERM_PROJECT/model/", exist_ok=True)
bert_model_name="uncased_L-12_H-768_A-12"
bert_ckpt_dir = ('home/ubuntu/TERM_PROJECT/model/uncased_L-12_H-768_A-12')
bert_ckpt_file = ('/home/ubuntu/TERM_PROJECT/model/uncased_L-12_H-768_A-12/bert_ckpt_dir/bert_model.ckpt')
bert_config_file = ("/home/ubuntu/TERM_PROJECT/model/uncased_L-12_H-768_A-12/bert_config_file/bert_config.json")

##input data prepp
class DataPreparation:
  text_column = ["patterns"]
  label_column = "tag"

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes

    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self.prepare_data, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self.data_padding, [self.train_x, self.test_x])

  def prepare_data(self, df):
    x, y = [], []

    for _, row in tqdm(df.iterrows()):
      text, label = row[DataPreparation.text_column], row[DataPreparation.label_column]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def data_padding(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)
tokenizer = FullTokenizer(vocab_file=('/home/ubuntu/TERM_PROJECT/model/uncased_L-12_H-768_A-12/bert_ckpt_dir/vocab.txt'))

#Creating a model:
def model_defination(max_seq_len, bert_ckpt_file):
  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
    bc = StockBertConfig.from_json_string(reader.read())
    bert_params = map_stock_config_to_params(bc)
    bert_params.adapter_size = None
    bert = BertModelLayer.from_params(bert_params, name="bert")

  input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))

  load_stock_weights(bert, bert_ckpt_file)

  return model

classes = train.tag.unique().tolist()
print(classes)
data = DataPreparation(train, test, tokenizer, classes, max_seq_len=128)
model = model_defination(data.max_seq_len, bert_ckpt_file)

print(model.summary())

model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

log_dir = "log/intent_classification/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
  x=data.train_x,
  y=data.train_y,
  validation_split=0.2,
  batch_size=128,
  shuffle=True,
  epochs=5,
  callbacks=[tensorboard_callback]
)

train_acc = model.evaluate(data.train_x, data.train_y)
test_acc = model.evaluate(data.test_x, data.test_y)


print("train acc", train_acc)
print("test acc", test_acc)