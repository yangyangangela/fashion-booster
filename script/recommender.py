"""
Recommender: based on the rental history of a given user, 
    recommend similar items using the item description similarity

Author: Yang Yang
Date: 2017-09-19
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pandas as pandas
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import enchant
import re
from nltk.stem.lancaster import LancasterStemmer
from sklearn.metrics.pairwise import linear_kernel

# Connection to DB
DBNAME = 'mylocaldb' 
USERNAME = 'yangyang'
CON = None
CON = psycopg2.connect(database = DBNAME, user = USERNAME)

# base for brand_matrix, df with tokens
BASE_PATH = "/Users/yangyang/Documents/Study/Insight/project/code/source_data"

# SQL query
ITEMS="""
SELECT id, description,item_type,size,brand
FROM items
"""

BRAND_RENTALS="""
SELECT rental_items.item_id, rental_items.rental_id, 
    rentals.renter_id, UPPER(items.brand) as brand
FROM rental_items
LEFT JOIN items
    ON items.id=rental_items.item_id
LEFT JOIN rentals
    ON rental_items.rental_id = rentals.id
"""

# Constants to correct brands
SEARCH_N_REPLACE= {'LAURENT':'YVES SAINT LAURENT',
                    'GABBANA':'DOLCE GABBANA',
                    'PORTRAIT':'SELF PORTRAIT',
                    'BCBG':'BCBG',
                    'LEMONS':'FOR LOVE AND LEMONS',
                    'ELIZABETH': 'ELIZABETH AND JAMES',
                    'JILL STUART':'JILL STUART',
                    'PARTNERS':'FAME AND PARTNERS',
                    'CREW':'J.CREW',
                    'JIMMY CHOO':'JIMMY CHOO',
                    'LOVER':'LOVER',
                    'ALEXANDER MCQUEEN':'ALEXANDER MCQUEEN',
                    'MICHAEL KORS':'MICHAEL KORS',
                    'MONIQUE LHUILLIER':'MONIQUE LHUILLIER',
                    'RACHEL ROY':'RACHEL ROY',                   
                    }


def __size_to_number(x):
    """
    Switch the non-float size to number.
    """
    SIZE_MAP={'One-Size':None,'XS':0,'S':4,'M':8,'L':12}
    try:
        return float(x)
    except:
        if x:
            return SIZE_MAP[x]
        else:
            return None


def __fix_brand(x):
    """
    Fix the typo in brands.
    """
    replace = x
    for kw in SEARCH_N_REPLACE.keys():
        if kw in x:
            replace = SEARCH_N_REPLACE[kw]
    return replace


def load_item_data():
    """
    Return the df with the following columns:
        id: item id
        description: original item description provided by users
        item_type: dress, shoes, bags, tops, etc.
        size: original size, numbers (8), charactre(XS/S), One-Size
        brand: original brand name

    """
    df = pd.read_sql_query(ITEMS, CON)

    df['description'] = df['description'].astype('str')
    df = df.set_index(['id'])

    # fix size and brand
    df['size_number'] = df['size'].apply(__size_to_number)
    df['brand'] = df['brand'].apply(__fix_brand)
    df = df[~df['brand'].isnull()]

    return df


def load_brand_data():
    """
    Return the df with the following columns: item_id, rental_id, renter_id,
    capitalized brand, where each row is a rented item record.
    """
    df = pd.read_sql_query(BRAND_RENTALS, CON)

    # remove none data in brand and renter_id
    df = df[(~df['brand'].isnull()) & (~df['renter_id'].isnull())]

    # fix brand
    df['brand'] = df['brand'].str.strip()
    df['brand'] = df['brand'].apply(__fix_brand)

    df = df[~df['brand'].isnull()]
    return df


def description_tokenizer(item_df):
    """
    Taken a series of item description, keep the following tokens:
        - transfer back to sinular
        - keep noun and adjective
        - 

    """
    # Only keep adv, adj, and noun
    keep_list = [
        'JJ',#  adjective   'big'
        'JJR',# adjective, comparative  'bigger'
        'JJS',# adjective, superlative  'biggest'
        'NN',#  noun, singular 'desk'
        'NNS',# noun plural 'desks'
        'NNP',# proper noun, singular   'Harrison'
        'NNPS',#    proper noun, plural 'Americans'
    ]


    eng_checker = enchant.Dict("en_US")# check english
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')# remove number
    lemmatizer = nltk.stem.WordNetLemmatizer()# plural to singular
    stop_words = set(stopwords.words('english'))
    st = LancasterStemmer()# trace the stem of words

    
    def tokenizer_pipline(x):
        # remove numbers
        x = re.sub(r'\d+','',x).lower()

        # remove punctuation
        tokens = tokenizer.tokenize(x)  
        tags = nltk.pos_tag(tokens)
    
        # keep noun, adj, adv, remove plural
        filtered_tokens =[lemmatizer.lemmatize(t[0]) for t in tags if t[1] in keep_list]

        # keep only the stem words
        #filtered_tokens =[st.stem(w) for w in filtered_tokens]
    
        # remove useless words
        useless_words = ['color', 'size','time','true','composition','com',
                'fit','length','item','product','copy','label','measurement',
                'kind','product','none','ab','able','add','ad','age','answer',
                'anywhere','cc','cm','dress','skirt','shoe','top','bottom','pants']
    
        # remove stop_words, non-english
        word_list = [ w for w in filtered_tokens if 
                    ((not w in stop_words) and eng_checker.check(w) and (not w in useless_words))]
    
        # convert word list to string
        str1 = ' '.join(str(e) for e in word_list)    
        return str1
    
    item_df['tokens'] = item_df['description'].apply(tokenizer_pipline)
    return item_df






class brand_similarity(object):
    """
    Brand similarity analysis

    """
    def __init__(self):
        self.data = load_brand_data()

    def top_brand_ls(self,thres=5):
        """
        Return the brand name list for occurence>5.
        """
        top_brands = self.data['brand'].value_counts()
        return top_brands[top_brands>thres].index.tolist()

    def brand_cooccur_matrix(self,thres=5):
        """
        Return the co_occurence matrix of brands, given the rentals>thres.
        """
        # build a matrix to save the co-occurence frequency of two brands
        brand_ls = self.top_brand_ls(thres)
        n = len(brand_ls)
        brand_occur_matrix = np.matrix((n,n))

        my_data = self.data[self.data.brand.isin(brand_ls)]

        le = preprocessing.LabelEncoder()
        le.fit(my_data['brand'])

        renter_ls = my_data.renter_id.unique().tolist()

        # Encode the brand to classes
        my_data['brand_code'] = le.transform(my_data['brand'])

        # iterate my_data, fill in the matrix
        n = len(le.classes_)
        A = np.zeros((n,n))

        for idx in renter_ls:
            items = my_data.ix[my_data.renter_id==idx,'brand_code'].values
            try:
                A[np.ix_(items,items)] += 1
            except:
                print(items)

        brand_list = le.classes_.tolist()
        brand_matrix = pd.DataFrame(A,index=brand_list,columns=brand_list)
        return brand_matrix



class customized_recommender(object):
    def __init__(self, item_df = None):
        """
        Initilize
        - the CR will automatically load item_df

        """
        if item_df:
            self.df = item_df
        else:
            token_path = BASE_PATH + "/item_df_n_tokens.csv"
            self.df = pd.read_csv(token_path,index_col=0)

        if not 'tokens' in self.df:
            self.df['tokens'] = description_tokenizer(item_df['description'])

        self.item = 0
        self.mask = None

    def __select_brands(self,the_brand,brand_number=10):
        """
        Return a T/F mask to select top brand_number related items.
        If the brand is not in brand_matrix, return the top popular brands.

        """
        BRAND_MATRIX_PATH = BASE_PATH + "/brand_matrix.csv"
        brand_cooccur_matrix = pd.read_csv(BRAND_MATRIX_PATH,index_col=0)
        
        if the_brand in brand_cooccur_matrix:
            bls = brand_cooccur_matrix[ the_brand ].nlargest(brand_number).index.tolist()
            bls.append(the_brand)
        
        else:
            bls = brand_cooccur_matrix.max().nlargest(brand_number).index.tolist()

        return self.df.brand.isin( bls )


    def __select_type_and_size(self,the_type,the_size):
        """
        Return a T/F mask to select correct type and size.
        """
        NO_SIZE = ['bags','accessories']    
        if the_type == 'shoes':
            vari = 1
        else:
            vari = 2
    
        # return the same type of cloths.
        if the_type in NO_SIZE:
            return self.df.item_type.isin(NO_SIZE)
        # return shoes or NO_SIZE
        elif the_type == 'shoes':
            mask = ( ((self.df.item_type=='shoes') &
                    (self.df.size_number>=the_size-vari) &
                    (self.df.size_number<=the_size+vari)) |
                    (self.df.item_type.isin(NO_SIZE))
                    )
        # return the type or NO_SIZE
        else:
            mask = ( ((self.df.item_type==the_type) &
                    (self.df.size_number>=the_size-vari) &
                    (self.df.size_number<=the_size+vari)) | 
                    (self.df.item_type.isin(NO_SIZE))
                    )

 
        return mask

    def filter_recommendation(self, given_item, type_n_size=True, brands=True, brand_number=3):
        """
        Return a T/F mask, w/ the same row as item_df.
        Call this function to 
            - pass the selected item to self.item.
            - set the self.mask to the recommended item pool for the selected item.
        """
        self.item = given_item
        mask = ~self.df['brand'].isnull()
        if type_n_size:
            the_type = self.df.loc[given_item,'item_type']
            the_size = self.df.loc[given_item,'size_number']
            mask = mask & self.__select_type_and_size(the_type,the_size)
        
        if brands:
            the_brand = self.df.loc[given_item,'brand']
            if the_brand:
                test_mask = mask & self.__select_brands(the_brand,brand_number=brand_number)

        # if not enough test_mask, remove the constraints on brand
        if sum(test_mask) < 5:
            self.mask = mask
            return mask
        else:
            self.mask = test_mask
            return test_mask


    def recommend_items(self,top=5):
        """
        Output the df of recommended items based on current self.item.

        """

        rdf = self.df[self.mask]

        # return all filtered stuff when no enough items to recommend.
        if len(rdf)<=top:
            return rdf.index.tolist()

        # calculate the similarity between items
        S = self.similarity(self.item, rdf.index.tolist())

        # find the item id of the top 5 colomns
        return rdf.loc[S['score'].nlargest(top).index.tolist(),:]



    def similarity(self,item, itemLs):
        """
        Return the similarity series between an item and an Itemlist.
        """
        count_vect = CountVectorizer()
        TOKEN_PATH = BASE_PATH + "/item_df_n_tokens.csv"

        if not item in itemLs:
            itemLs.append(item)

        # load tokens
        df = pd.read_csv(TOKEN_PATH,index_col=0)

        # only select the items in list
        data = df.loc[itemLs,:]
        data['tokens'] = data['tokens'].fillna('.')

        # cosine similarity on tf_idf counts
        X_train_counts = count_vect.fit_transform(data['tokens'])
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        ind = data.index.tolist().index(item)
        cosine_similarities = linear_kernel(X_train_tfidf[ind], X_train_tfidf)[0]

        return pd.DataFrame(cosine_similarities,index=data.index,columns=['score'])















