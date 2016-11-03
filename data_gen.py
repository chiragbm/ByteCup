'''
Created on Oct 29, 2016

@author: chira
'''
import numpy
import pandas
from pandas.core.series import Series
from sklearn.linear_model.logistic import LogisticRegression

def get_mean_median(train_x):
    
    mean = numpy.mean(train_x, axis = 0)
    std = numpy.std(train_x, axis=0, ddof = 1)
    
    return mean, std

def normalize_data(d_x, col):
    
    train_x = [d[col] for d in d_x]
    
    mean, std = get_mean_median(train_x)
    
    for d in d_x:
        d[col] = (d[col] - mean) / std
        


def process_data(data, prefix):
    
    tagset = list()
    for tag in data['tags']:
        tags = str(tag).split('/')
        for t in tags:
            tagset.append(t)
    
    tagset = set(tagset)
    
    for l in tagset:
        data[prefix + str(l)] = Series([0 for _ in range(len(data['tags']))], index = data.index)
      
    for i in range(len(data['tags'])):
        tags = str(data['tags'][i]).split('/')
        for tag in tags:
            data.loc[i, prefix + tag] = 1
#         
    del data['tags']
    del data['wid']
    del data['cid'] 
    
####################3 ------------Start -------------#########################
user_file_name = "user_info.txt"

user_data = {}
user_data = pandas.read_csv(user_file_name, sep = '\t',header=None, names = ["uid","tags","wid","cid"])

process_data(user_data, 'u')

ques_file_name = "question_info.txt"
ques_data = {}
ques_data = pandas.read_csv(ques_file_name, sep = '\t',header=None, names = ["qid","tags","wid","cid","upvotes","answers","top_quality_ans"])

process_data(ques_data, 'q')


'''
################################# Train Data $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
'''

f = open('training_data.txt', 'w')

train_file = 'invited_info_train.txt'
train_data = pandas.read_csv(train_file,  sep = '\t', header=None, names = ["qid","uid","label"])
train_data.drop_duplicates(subset = ['qid', 'uid'], keep = 'first', inplace = True)

df_intermediate = pandas.merge(train_data, user_data, how='inner', on=['uid'], sort= False)
df_train = pandas.merge(df_intermediate, ques_data, how='inner', on=['qid'], sort= False)

train_y = []
train_x = []

for i in range(len(df_train['label'])):
    
    inner= []
    
    train_y.append(df_train['label'][i])
    
    for j in range(143):
        inner.append(df_train['u'+str(j)][i])
    
    for j in range(20):
        inner.append(df_train['q'+str(j)][i])
        
    inner.append(df_train['upvotes'][i])
    inner.append(df_train['answers'][i])
    inner.append(df_train['top_quality_ans'][i])
    inner.append(df_train['label'][i])
    
    
    f.write(','.join(map(str, inner)))
    f.write('\n')
    
    train_x.append(inner)


# normalize_data(train_x, 163)
# normalize_data(train_x, 164)
# normalize_data(train_x, 165)


'''

# numpy.array(train_y).reshape(-1, 1)
 
model = LogisticRegression()
model.fit(train_x, train_y)
'''

########################### - Test Data $$$$$$$$$$$$$$$$$$$$$$$$$$$$4

'''

     
test_data = {}
test_data = pandas.read_csv('validate_nolabel.txt',  sep = ',', names = ["qid","uid","label"])

ques = [] 
 
df_intermediate_test = pandas.merge(test_data, user_data, how='inner', on=['uid'], sort= False)
df_test = pandas.merge(df_intermediate_test, ques_data, how='inner', on=['qid'], sort= False)
 
# del df_test['qid']
# del df_test['uid']
 
test_x = []
 
for i in range(1000):
     
    inner= []
    q = []
    
    uid = df_test['uid'][i]
    qid = df_test['qid'][i]
    q.append(qid)
    q.append(uid)
    ques.append(q)
     
    for j in range(143):
        inner.append(df_test['u'+str(j)][i])
     
    for j in range(20):
        inner.append(df_test['q'+str(j)][i])
         
    inner.append(df_test['upvotes'][i])
    inner.append(df_test['answers'][i])
    inner.append(df_test['top_quality_ans'][i])
     
    test_x.append(inner)


out = open('tempp.csv', 'w')
out.write('qid,uid,label\n')

for i in range(len(test_x)):
       
    out.write('%s,%s,%s\n'%(ques[i][0],ques[i][1],model.predict_proba(test_x[i])[0][0]))
  
out.close()
'''
