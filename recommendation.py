from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning)

# load product information
df_products = pd.read_csv('jcpenney-products_subset.zip',index_col=0)

# get product descriptions as our dataset
X_products = df_products.description
len(X_products)


# instantiate a TfidfVectorizer with
tfidf = TfidfVectorizer(min_df=5, stop_words='english')

# instantiate a TfidfVectorizer with
X_tfidf = tfidf.fit_transform(X_products)
X_tfidf.shape


feature_names = tfidf.get_feature_names()
print(feature_names[:10])
print(feature_names[-10:])

lda = LatentDirichletAllocation(n_components=20, n_jobs=-1, random_state=123)

X_lda = lda.fit_transform(X_tfidf)
X_lda.shape

# a utility function to print out the terms that are highly likely for each topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
          
print_top_words(lda,feature_names,10)




from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(X_lda)
similarities.shape

query_idx = 500

df_products.iloc[query_idx]['name_title']

# get the similarity scores for our query_idx
query_scores = similarities[query_idx,:]

best_sorted_asc = np.argsort(query_scores)
best_sorted_asc[-10:]

best_sorted_desc = best_sorted_asc[::-1]
best_sorted_desc[:10]

for i in best_sorted_desc[:10]:
    print(df_products.name_title[i])