import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler,PowerTransformer
import warnings as wn
from random import sample
from sklearn.model_selection import train_test_split as tts
# import VDN as vd
import connectorx as cx 
import dotenv 
dotenv.load_dotenv()
wn.filterwarnings('ignore')


# 個股每日成交資訊(DIOIS)(Daily information of individual stock)
# 個股日本益比、殖利率及股價淨值比(DEDOIS)(Daily EPS Dividend yield of individual stocks)
# 投信買賣超(Sdifference)
# 外資及陸資買賣超總匯(Fdifference)
# 三大法人買賣超(investorDifference)
class datascience:
    def __init__(self):
        self.train_x_final = []
        self.train_y_final = []
        self.test_x_final = []
        self.test_y_final = []
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.one = None
        self.label = None
        self.df = None
        self.alltrain_x = None
        self.alltrain_y = None
        self.allx = None
        self.ally = None
        self.train_day = 7

    def DII_analysis(self, stock,model_exist=False):
        
        iterate_count = range(1)
        if model_exist ==True:
            iterate_count = range(1)
        df = cx.read_sql(f"postgres://{os.getenv('psql_user')}:{os.getenv('psql_pswd')}@{os.getenv('psql_host')}/{os.getenv('psql_database')}","SELECT * FROM price")  # "
        df = df[df["stock_id"]==f'{stock}']
        self.df = df
        train_y, train_y_one_hot, self.one, self.label = self.train_y_create(df)
        df = df.join(train_y)
        train_x = df.drop(["train_y", "date", "stock_id", "最後揭示買價", "最後揭示買價", "最後揭示賣價", "最後揭示賣量", "最後揭示買量", "本益比",'漲跌價差'],axis=1)
        train_x = train_x[:int(len(train_x)*0.7)]
        test_x = train_x[int(len(train_x)*0.7):]
        train_y = train_y[:int(len(train_x)*0.7)]
        train_x= self.datareshape(train_x)
        test_x = self.datareshape(test_x)
        train_x, train_y = self.datafinalprocess(train_y_one_hot,
                                                train_x,
                                                train_y)
        for count in iterate_count:
            self.train_x.append(train_x[:])
            self.train_y.append(train_y[:])
        for count in iterate_count:
            self.test_x.append(test_x)
        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)
        data = {"train_x":train_x,
                "train_y":train_y,
                "test_x":test_x,
                "onehot":self.one,
                "label":self.label,
                "df":self.df}
        return data 
            

    def datareshape(self,train_x):
        print("data reshape .....")
        train_x_arr_all = []
        day = self.train_day
        scale = MinMaxScaler()
        Powerscaler = PowerTransformer(method="yeo-johnson")
        train_x_arr_all = []
        for col in train_x.columns:
            train_x_col = np.array(train_x[col].iloc[:]).reshape(-1,1)            
            train_x[col] = scale.fit_transform(train_x_col)
            train_x_col = np.array(train_x[col].iloc[:]).reshape(-1,1)            
            train_x[col] = Powerscaler.fit_transform(train_x_col)

        for i in range(day,len(train_x)):
            train_x_conv = train_x.iloc[i-day:i].to_numpy()
            train_x_arr_all.append(np.array(train_x_conv))
        train_x_arr_all = np.array(train_x_arr_all)
        
        return train_x_arr_all 
    @staticmethod
    def add_ma_df(df):
        for i in range(5, len(df)):
            df.loc[i, "5ma"] = df["收盤價"][i - 5:i].astype("float").mean()
        df.loc[:, "5ma"] = df["5ma"].fillna(0)
        for i in range(10, len(df)):
            df.loc[i, "10ma"] = df["收盤價"][i - 10:i].astype("float").mean()
        df.loc[:, "10ma"] = df["10ma"].fillna(0)
        return df

    def datafinalprocess(self, train_y, train_x, index_y):
        print("data final process ")
        # let the 0 1 -1 have equal mount for  training have equal mount
        day = self.train_day
        train_y = train_y[day:]
        index_y = index_y[day:]
        index_y_buy = index_y[index_y["train_y"] == 1].index.tolist()
        index_y_sell = index_y[index_y["train_y"] == -1].index.tolist()
        index_y_equal = index_y[index_y["train_y"] == 0].index.tolist()
        equal_x = []
        buy_x = []
        sell_x = []
        if len(index_y_buy) > len(index_y_sell):
            sample_mount = len(index_y_sell)
        else:
            sample_mount = len(index_y_buy)
        buy_x = np.array([train_x[index - day] for index in index_y_buy[:sample_mount]])#[:-3]
        sell_x = np.array([train_x[index - day] for index in index_y_sell[:sample_mount]])
        equal_x =np.array([train_x[index - day] for index in index_y_equal[:sample_mount]])
        train_x = np.concatenate((buy_x, sell_x))
        train_x = np.concatenate((train_x, equal_x))
        buy_y = [train_y[index - day] for index in index_y_buy[:sample_mount]]#
        sell_y = [train_y[index - day] for index in index_y_sell[:sample_mount]]
        equal_y = [train_y[index - day] for index in index_y_equal[:sample_mount]]
        train_y = np.concatenate((buy_y, sell_y))
        train_y = np.concatenate((train_y, equal_y))
        print("i'm train_x shape",train_x.shape)
        
        return train_x, train_y

    def gray_picture(self, train_xs, train_ys):
        # train_ys = np.array(train_ys)[0]
        for x, y, name in zip(train_xs, train_ys, range(len(train_ys))):

            i = x.reshape(-1, 11, 11)
            gdata = i[0] * 255
            im = Image.fromarray(np.array(gdata).astype(np.uint8))
            im = im.convert("L")
            path = "../gray_picture/"
            if not os.path.exists("../gray_picture/1"):
                os.mkdir("../gray_picture/1")
            if not os.path.exists("../gray_picture/0"):
                os.mkdir("../gray_picture/0")
            if not os.path.exists("../gray_picture/-1"):
                os.mkdir("../gray_picture/-1")
            if np.array_equal(y, np.array([0, 0, 1])):
                im.save(path + "1/" + str(name) + ".jpg")
            elif np.array_equal(y, np.array([0, 1, 0])):
                im.save(path + "0/" + str(name) + ".jpg")
            elif np.array_equal(y, np.array([1, 0, 0])):
                im.save(path + "-1/" + str(name) + ".jpg")

    def train_y_create(self, df=None):
        print("train Y create ....")
        num = df["收盤價"].to_numpy().astype(float)
        train_y = np.zeros(num.shape, dtype=int)
        hstock = 0
        day =self.train_day
        # could estimate performance and optimise algorithm -> structure first
        for today in range(0, num.shape[0] - day):
            # current idea is if today isn't the extremum  day will not update the value
            # -1 present sale 1 present buy
            monthmax = np.max(num[today+1:day + today])
            monthmin = np.min(num[today+1:day + today])

            if (num[today] < monthmin) :
                if hstock <= 0:
                    hstock += 1
                    train_y[today] = 1
            elif (num[today] > monthmax) :
                if hstock >= 1:
                    hstock -= 1
                    train_y[today] = -1
        label_enconder = LabelEncoder()
        train_y_bny = label_enconder.fit_transform(train_y)
        train_y_bny = train_y_bny.reshape(-1, 1)
        onehotencoder = OneHotEncoder()

        train_y_one_hot = onehotencoder.fit_transform(train_y_bny).toarray()
        if not os.path.exists("../training"):
            os.mkdir("../training")
            self.savefile("onehotencoder", onehotencoder)
            self.savefile("labelencoder", label_enconder)

        train_y = pd.DataFrame(train_y, columns=["train_y"])
        train_y_one_hot = np.array(train_y_one_hot[:])
        return train_y, train_y_one_hot, onehotencoder, label_enconder


if __name__ == "__main__":
    dc = datascience()
    dc.DII_analysis("2303")
