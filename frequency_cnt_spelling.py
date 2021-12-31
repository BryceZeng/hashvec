from polyfuzz import PolyFuzz
import textdistance
import sklearn.cluster
from scipy import spatial
from fuzzywuzzy import fuzz
import io
import numpy as np
from symspellpy import SymSpell, Verbosity
from itertools import islice
import pandas as pd
import re
train = pd.read_csv('train/negative.csv', encoding='latin1')

replaceword = {'djalan': 'jalan', 'jl': 'jalan', 'djl': 'jalan', 'dalan': 'jalan',
               'jln': 'jalan', 'jn': 'jalan', 'gg': 'gang',
               'g g': 'gang', 'blk': 'blok', 'block': 'blok',
               'number': 'no'
               }
pattern = r'\b({})\b'.format(
    '|'.join(sorted(re.escape(k) for k in replaceword)))


def cleanString(string):
    string = str(string).lower()
    string = string.replace('\r', ' ').replace('\n', ' ')
    string = re.sub(pattern, lambda m: replaceword.get(
        m.group(0)), string, flags=re.IGNORECASE)
    string = re.sub(r'[^\w\s]', ' ', string)
    string = re.sub('\d+', ' ', string)
    string = re.sub('\s+', ' ', string)
    string = string.strip()
    return string


train['detail_list'] = train.apply(
    lambda row: cleanString(row['detail_list']), axis=1)

density = train.detail_list.str.split(expand=True).stack().value_counts()

density[density > 1000].sort_values(
    ascending=False).to_csv('density.txt', sep=' ')

d1 = density.to_dict()
d1 = density.index.tolist()
d1 = {i: 1 for i in d1}
d1 = train['detail_list'].tolist()

# textfile = open("single_word.txt", "w")
# for element in d1:
#     textfile.write(element + "\n")
# textfile.close()

# from spello.model import SpellCorrectionModel
# sp = SpellCorrectionModel(language='en')
# sp.load('spello_model/model.pkl')
# sp.train(d1)
# input = 'jjalan ganngg haji jjamal 56(rt008/003)'
# sp.spell_correct(input)
# sp_dict = sp.spell_correct(input).get('correction_dict')
# # sp_dict = {v: k for k, v in sp_dict.items()}
# for word, initial in sp_dict.items():
#     input = input.replace(word.lower(), initial)
# print(input)

# sp.save(model_save_dir='spello_model')

'''
textblob
'''

# from textblob import TextBlob
# from textblob.en import Spelling
# path = "density.txt"
# spelling = Spelling(path=path)
# # Here, 'names' is a list of all the 1,000 correctly spelled names.
# # e.g. ['Liam', 'Noah', 'William', 'James', ...
# spelling.train(path)
# spelling.suggest('pesantren jalann gangg daarul mansur')[0][0]

'''
enchant_dict_add
'''

# import enchant
# pwl = enchant.request_pwl_dict("single_word.txt")
# # d2 = enchant.DictWithPWL("single_word.txt")
# for word in input.split():
#     if word not in pwl:
#         print(word)
#     print(' '.join(pwl.suggest(word)[0]))

'''
SymSpell
'''

sym_spell = SymSpell()
path = "density.txt"
sym_spell.load_dictionary(path, 0, 1)

# Print out first 5 elements to demonstrate that dictionary is
# successfully loaded
print(list(islice(sym_spell.words.items(), 5)))


def spelling_correct(word):
    # Look up the spelling correction
    correction_info = sym_spell.lookup(
        word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
    # If a correction exists, output the corrected word
    if correction_info:
        return correction_info[0].term
    # Otherwise, output the original word
    else:
        return word


result = ''
for word in input.split():
    corrected = spelling_correct(word)
    result += str(corrected) + " "
print(result)


'''
Grouping words together based on their spelling
'''


s = """Name           ID            Value
0    James           1             10
1    James 2         2             142
2    Bike            3             1
3    Bicycle         4             1197
4    James Marsh     5             12
5    Ants            6             54
6    Job             7             6
7    Michael         8             80007  
8    Arm             9             47 
9    Mike K          10            9
10   Michael k       11            1"""
df = pd.read_csv(io.StringIO(s), sep='\s\s+', engine='python')

'''
affprop
'''
# names = df.Name.values
# sim = spatial.distance.pdist(names.reshape(
#     (-1, 1)), lambda x, y: fuzz.WRatio(x, y))
# affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed",damping = 0.9,  random_state=None)
# affprop.fit(spatial.distance.squareform(sim))

# df['cluster'] = affprop.labels_
# df['cluster_score'] = affprop.affinity_matrix_.max(1)


'''
StringGrouper
'''
# import pandas as pd
# import numpy as np
# from string_grouper import match_strings, match_most_similar, \
# 	group_similar_strings, compute_pairwise_similarities, \
# 	StringGrouper

# # Group customers with similar names:
# df[["group-id", "Name"]] = \
#     group_similar_strings(df["Name"], min_similarity=0.5)
# # Display the mapping table:
# df
