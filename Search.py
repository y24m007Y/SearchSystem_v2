import numpy as np
import pandas as pd
import fugashi
import unidic
import re
import json
import string
from pathlib2 import Path
from rank_bm25 import BM25Okapi
import torch
import argparse
import os, sys
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer, models
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Searcher():
    def __init__(self, path="DataBase/default"):
        filepath = Path(path)
        filenames = [filename for filename in filepath.iterdir() if re.search("csv", filename.name)]
        df = None
        for filename in filenames:
            tmp_df = pd.read_csv(filename, encoding='utf-8')

            if df is None:
                df = tmp_df
            else:
                df = pd.concat((df, tmp_df), axis=0)
        df = df.dropna()
        df = df.reset_index()
        print(df.shape)
        self.set_data(df)
    
    def set_data(self, data):
        self.url = data.url
        self.title = data.title
        self.content = data.body
        self.tags = data.tags.unique()
        return
        
    def simulate(self, query):
        return query
    
    def search(self):
        query = input("検索:")
        scores = self.simulate(query)
        rank = np.argsort(scores)[::-1]
        rank_10 = [int(rank[i]) for i in range(10)]
        print("{}:検索結果".format(query))
        for i in range(len(rank_10)):
            if scores[rank_10[i]] > 0:
                print(f"Rank{i+1} タイトル: {self.title[rank_10[i]]} \n url: {self.url[rank_10[i]]}")
            else:
                break
        return 
    
class bm25(Searcher):
    def __init__(self, path="DataBase/default"):
        super().__init__(path)
        self.pattern = "!\#$%&\'\\\\()*+,./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％～■"
        with open("custom_stopwords_ja.json", 'r') as f:
            jsondata = json.load(f)
        self.stopwords = jsondata['stopwords']
        self.tagger = fugashi.Tagger(f"-Owakati {unidic.DICDIR}")
        self.get_model(self.content)
        
    def preprocess(self, text):
        text = re.sub('\n', '', text)
        text = re.sub(r'[{}]'.format(self.pattern), '', text)
        sub_words = self.tagger.parse(text)
        sub_words = [word for word in sub_words.split() if word not in self.stopwords]
        return sub_words

    def get_model(self, texts):
        texts = texts.apply(lambda x: self.preprocess(x))
        texts = texts.to_list()
        print(len(texts))
        self.bm = BM25Okapi(texts)
        if bool(bm25):
            print("モデルの構築ができました")
            print(self.bm)
        else:
            print("モデルの構築ができませんでした")
            return 
    
    def simulate(self, query):
        tokenized_query = self.preprocess(query)
        scores = self.bm.get_scores(tokenized_query)
        return scores

class laai(Searcher):
    def __init__(self, path="DataBase/default"):
        super().__init__(path)
        MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
        bert = models.Transformer(MODEL_NAME)
        pooling = models.Pooling(
        bert.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        )
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        device = 'cpu'
        print(device)
        self.model = SentenceTransformer(modules=[bert, pooling], device=device)
        self.model.load_state_dict(torch.load('models/after-pre-train3(version2).pth'))

        print("モデルのロードができました")

        filepath = "corpus_vector/after-qiita-train2(version2)3000.pkl"

        if os.path.exists(filepath):
            self.corpus_vec = pd.read_pickle(filepath)
        else:
            print("ベクトルコーパスがありません")
            sys.exit()
        self.corpus_vec = self.corpus_vec.dropna()
        self.corpus_vec = self.corpus_vec.drop(columns="tags")
        print(self.corpus_vec.shape)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
        )
            
    def get_embedding(self, text):
        texts = self.text_splitter.split_text(text)
        result = 0
        for text in texts:
            text_vec = self.model.encode(text)
            if result == 0:
                result = text_vec
            else:
                result += text_vec
        return result / len(texts)
    
    def get_score(self, query):
        score = np.zeros(len(self.corpus_vec))
        for id in self.corpus_vec.index:
            vec = self.corpus_vec.iloc[id,:].to_numpy()
            score[id] = query @ vec.T / np.sqrt((query@query.T)*(vec@vec.T))
        return score
    
    def simulate(self, query):
        query_vec = self.get_embedding(query)
        scores = self.get_score(query_vec)
        return scores

class hibrid(Searcher):
    def __init__(self, path="DataBase/default"):
        super().__init__(path)
        self.bm = bm25(path)
        self.labse = laai(path)

    def get_score(self, score_k, score_l, difficulty, k):
        scores = np.zeros(len(score_k))
        for i in range(len(score_k)):
            score = (1 / (k+score_k[i]+1)) + (1 / (k+score_l[i]+1)) + (1/ (k+difficulty[i]+1))
            scores[i] += score
        return scores

    #それぞれの検索結果による各文書ごとの順位付け
    def get_rank_index(self, anc_score):
        result = dict()
        for i in range(len(anc_score)):
            result[anc_score[i]] = i
        return result

    #単語頻度による各文書ごとの順位付け
    def neary_cat(self, query):
        difficulty_path = Path("Difficulty/word_value") #スコアが最も高いタグがクエリと関連性の高いタグとする
        scoringfiles = [file for file in difficulty_path.iterdir()]
        tagger = fugashi.Tagger(f"{unidic.DICDIR}")
        words = [word.feature.lemma if not re.search("[a-zA-Z]+", word.surface) else word.surface.lower() for word in tagger(query)]
        cats = [re.findall('(?<=_).+(?=\.)', scoreingfile.name)[0] for scoreingfile in scoringfiles]
        check_cat = defaultdict(float)
        for i in range(len(scoringfiles)):
            scoringfile = scoringfiles[i]
            check_cat[self.tags[i]] = 0
            with open(scoringfile, mode='rb') as f:
                wordvalue = pickle.load(f)
                for word in words:
                    if word in wordvalue.keys():
                        check_cat[cats[i]] += wordvalue[word]
        cat = max(check_cat.items(), key=lambda x: x[1])
        return cat[0]
    
    def get_difficulty_score(self, query):
        nearest_cat = self.neary_cat(query)
        scoringfilepath = Path(f"Difficulty/difficulty_score/{nearest_cat}.pkl")
        difficulty_df = pd.read_pickle(scoringfilepath)
        return difficulty_df.index
        

    def simulate(self, query, k=60):
        keyword_score = self.bm.simulate(query)
        lm_score = self.labse.simulate(query)
        keyword_rank = np.argsort(keyword_score)[::-1]
        lm_rank = np.argsort(lm_score)[::-1]
        keyword_rank = self.get_rank_index(keyword_rank)
        lm_rank = self.get_rank_index(lm_rank)
        difficulty_rank = self.get_difficulty_score(query)
        del keyword_score, lm_score
        scores = self.get_score(keyword_rank, lm_rank, difficulty_rank, k)
        return scores

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bm25", 
                         help="press -b keyword", action="store_true")
    parser.add_argument("-l", "--laai", help="press -l vector", action="store_true")
    parser.add_argument("-hi", "--hibrid", help="press -hi hibrid", action="store_true")
    args = parser.parse_args()
    if args.bm25:
        searcher = bm25()
    elif args.laai:
        searcher = laai()
    elif args.hibrid:
        searcher = hibrid()
    searcher.search()


        
        
        
    