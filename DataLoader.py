import os
import json
import random
import numpy as np
import re
import tensorflow_hub as hub
import tensorflow as tf
import pickle
import csv
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self):
        self.batch_size = 128
        self.long_seq = 5
        self.embed_size = 512
        self.max_daily_tweets = 282

        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.embed = hub.load(module_url)
        print ("module %s loaded\n" % module_url)

        self.price_data_path = 'data/price/preprocessed'
        self.tweet_data_path = 'data/tweet/preprocessed'

        self.fnames = [fname for fname in os.listdir(self.price_data_path) if
                  os.path.isfile(os.path.join(self.price_data_path, fname))]
        print(len(self.fnames), 'tickers selected')

        single_ticker = np.genfromtxt(os.path.join(self.price_data_path, 'AAPL.txt'), dtype=str, skip_header=False)
        self.trading_dates = single_ticker[:, 0][::-1].tolist()
        print(len(self.trading_dates), 'trading dates:')

        dates_index = {}
        date_format='%Y-%m-%d'
        for index, date in enumerate(self.trading_dates):
            dates_index[date] = index

        # dates taken from baseline works
        self.tra_ind = dates_index['2014-01-02']
        self.val_ind = dates_index['2015-08-03']
        self.tes_ind = dates_index['2015-10-01']
        self.end_ind = dates_index['2016-01-04']

        print(self.tra_ind, self.val_ind, self.tes_ind, self.end_ind)


    def get_price_data(self, tic_ind, date_ind):
        filename = 'datapoints/pricedata_' + str(tic_ind) + '_' + str(date_ind)

        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                price_data = np.load(f)

        else:
            single_EOD = np.genfromtxt(os.path.join(self.price_data_path, self.fnames[tic_ind]), dtype=float, skip_header=False)
            single_EOD = single_EOD[:, 1:][::-1, :]
            seq_EOD = single_EOD[date_ind - self.long_seq: date_ind, :]
            price_data = np.concatenate((seq_EOD[:, 0][:, np.newaxis], seq_EOD[:, 2:4]), axis=1)

            with open(filename, 'wb') as f:
                np.save(f, price_data)

        return price_data


    def get_text_data(self, tic_ind, date_ind):
        filename = 'datapoints/textdata_' + str(tic_ind) + '_' + str(date_ind)

        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                stock_tweets = np.load(f)

        else:
            date_format = '%Y-%m-%d'
            ss = self.fnames[tic_ind][:-4]

            for ind in range(date_ind - self.long_seq, date_ind):
                sdate = datetime.strptime(self.trading_dates[ind - 1], date_format) + timedelta(days=1)
                edate = datetime.strptime(self.trading_dates[ind], date_format)

                if sdate == edate:
                    datelist = [edate.strftime(date_format)]
                else:
                    datelist = [(sdate+timedelta(days=x)).strftime(date_format) for x in range((edate-sdate).days + 1)]

                missing_tweets = True

                for date in datelist:
                    if os.path.isfile(os.path.join(self.tweet_data_path, ss, date)):
                        missing_tweets = False
                        break

                if missing_tweets:
                    raise Exception('Missing Tweet Data')

            stock_tweets = []

            for ind in range(date_ind - self.long_seq, date_ind):
                daily_tweets = []

                sdate = datetime.strptime(self.trading_dates[ind - 1], date_format) + timedelta(days=1)
                edate = datetime.strptime(self.trading_dates[ind], date_format)

                if sdate == edate:
                    datelist = [edate.strftime(date_format)]
                else:
                    datelist = [(sdate+timedelta(days=x)).strftime(date_format) for x in range((edate-sdate).days + 1)]

                for date in datelist:
                    if os.path.isfile(os.path.join(self.tweet_data_path, ss, date)):
                        f = open(os.path.join(self.tweet_data_path, ss, date))

                        for line in f:
                            tweet_words = json.loads(line)['text']
                            tweet = ' '.join(tweet_words)
                            # Process tweets: Drop all punctuations (except $)
                            tweet = re.sub(r'(?!<\d)\.(?!\d)|(?!<\d)\,(?!\d)|(?!<\d)\:(?!\d)|(?!<\d)\/(?!\d)|[^\w$.,:/]|\s+', ' ', tweet)
                            tweet = re.sub(r'\s+', ' ', tweet)
                            tweet = tweet.replace("$ ", "$")

                            if tweet not in daily_tweets:
                                daily_tweets.append(tweet)

                embedded_tweets = self.embed(daily_tweets)
                embedded_tweets = tf.pad(embedded_tweets, \
                                    tf.constant([[0, self.max_daily_tweets-embedded_tweets.shape[0],], [0, 0]]), mode='CONSTANT', constant_values=0)

                stock_tweets.append(embedded_tweets)

            with open(filename, 'wb') as f:
                np.save(f, stock_tweets)

        return stock_tweets


    def get_labels(self, tic_ind, date_ind):
        filename = 'datapoints/labels_' + str(tic_ind) + '_' + str(date_ind)

        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                labels = np.load(f)

        else:
            single_EOD = np.genfromtxt(os.path.join(self.price_data_path, self.fnames[tic_ind]), dtype=float, skip_header=False)
            single_EOD = single_EOD[:, 1:][::-1, :]

            close = single_EOD[date_ind][0]

            if close >= -0.005 and close <= 0.0055:
                raise Exception('Negligible Change')
            elif close > 0.0055:
                labels = [close, 0, 1]
            else:
                labels = [close, 1, 0]

            with open(filename, 'wb') as f:
               np.save(f, labels)

        return labels


    def get_indices(self):
        filename = 'indices.pkl'

        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                train_indices = pickle.load(f)
                valid_indices = pickle.load(f)
                test_indices = pickle.load(f)

        else:
            # get indices
            train_indices = []
            valid_indices = []
            test_indices = []

            for type in ["train", "valid", "test"]:
                indices = []
                total_ind = 0

                if type == "train":
                    start_ind = self.tra_ind
                    last_ind = self.val_ind
                elif type == "valid":
                    start_ind = self.val_ind
                    last_ind = self.tes_ind
                elif type == "test":
                    start_ind = self.tes_ind
                    last_ind = self.end_ind

                date_ind = start_ind
                tic_ind = 0

                while tic_ind < len(self.fnames):
                    if date_ind < self.long_seq:
                        continue
                    else:
                        try:
                            # Checking for validity one time, so do not need to check each epoch
                            self.get_price_data(tic_ind, date_ind)
                            self.get_text_data(tic_ind, date_ind)
                            self.get_labels(tic_ind, date_ind)

                            indices.append((tic_ind, date_ind))
                            total_ind += 1
                        except:
                            pass

                        date_ind += 1
                        if (date_ind >= last_ind):
                            tic_ind += 1
                            date_ind = start_ind

                    if tic_ind >= len(self.fnames):
                        break

                if type == "train":
                    train_indices = indices.copy()
                    print('train', len(train_indices))
                elif type == "valid":
                    valid_indices = indices.copy()
                    print('valid', len(valid_indices))
                elif type == "test":
                    test_indices = indices.copy()
                    print('test', len(test_indices))

            with open(filename, 'wb') as f:
                pickle.dump(train_indices, f)
                pickle.dump(valid_indices, f)
                pickle.dump(test_indices, f)

        return train_indices, valid_indices, test_indices


    def get_batch(self, type):
        train_indices, valid_indices, test_indices = self.get_indices()

        if type == "test":
            indices = test_indices.copy()

            price_data = np.zeros([len(indices), self.long_seq, 3], dtype=np.float32)
            text_data = np.zeros([len(indices), self.long_seq, self.max_daily_tweets, self.embed_size], dtype=np.float32)
            labels = np.zeros([len(indices), 3], dtype=np.float32)

            index = 0

            while index < len(indices):
                price_data[index] = self.get_price_data(*indices[index])
                text_data[index] = self.get_text_data(*indices[index])

                temp_labels = self.get_labels(*indices[index])
                labels[index]  = temp_labels

                index += 1

            yield([price_data, text_data, labels], labels)


        else:
            if type == "train":
                indices = train_indices.copy()
                random.shuffle(indices)
            elif type == "valid":
                indices = valid_indices.copy()

            price_data = np.zeros([self.batch_size, self.long_seq, 3], dtype=np.float32)
            text_data = np.zeros([self.batch_size, self.long_seq, self.max_daily_tweets, self.embed_size], dtype=np.float32)
            labels = np.zeros([self.batch_size, 3], dtype=np.float32)

            index = 0

            while index < len(indices):
                batch_ind = 0

                while batch_ind < self.batch_size:
                    price_data[batch_ind] = self.get_price_data(*indices[index])
                    text_data[batch_ind] = self.get_text_data(*indices[index])

                    temp_labels = self.get_labels(*indices[index])
                    labels[batch_ind] = temp_labels

                    batch_ind += 1
                    index += 1

                    if index >= len(indices):
                        if type == "test":
                            break
                        else:
                            if type == "train":
                                random.shuffle(indices)
                            index = 0

                yield([price_data, text_data, labels], labels)
