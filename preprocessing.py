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

    def DII_analysis(self, stock,model_exist=False):
        iterate_count = range(1)
        print(stock)
        def datareshape(train_x):
            print("data reshape .....")
            train_x_arr_all = []
            day = 10
            scale = MinMaxScaler()
            Powerscaler = PowerTransformer(method="yeo-johnson")
            
            for col in train_x.columns:
                train_x_col = np.array(train_x[col].iloc[:]).reshape(-1,1)            
                train_x[col] = scale.fit_transform(train_x_col)
                train_x_col = np.array(train_x[col].iloc[:]).reshape(-1,1)            
                train_x[col] = Powerscaler.fit_transform(train_x_col)


            train_x_arr_all = np.array(train_x.iloc[:])
            count = int(((train_x_arr_all.shape[0]*train_x_arr_all.shape[1])/8)//day)
            train_x_arr_all.resize((count,day,8))
            
            return train_x_arr_all 
        if model_exist ==True:
            iterate_count = range(1)
        day = 7
        print(os.getenv("psql_host"))
        df = cx.read_sql(f"postgres://{os.getenv('psql_user')}:{os.getenv('psql_pswd')}@{os.getenv('psql_host')}/{os.getenv('psql_database')}",
            "SELECT * FROM price")  # "
        print(stock)
        df = df[df["stock_id"]==f'{stock}']
        train_y, train_y_one_hot, self.one, self.label = self.train_y_create(df)
        df = df.join(train_y)
        self.df = df
        train_x = df.drop(["train_y", "date", "stock_id", "最後揭示買價", "最後揭示買價", "最後揭示賣價", "最後揭示賣量", "最後揭示買量", "本益比",'漲跌價差'],
                                   axis=1)

        # datapersent = int(train_x_original.shape[0])
        # data reshape

        train_x = train_x[:int(len(train_x)*0.7)]
        test_x = train_x[int(len(train_x)*0.7):]
        train_x= datareshape(train_x)
        test_x = datareshape(test_x)
        # test_x, test_y = self.datafinalprocess(test_y_bny,
        #                                        test_x_arr,
        #                                        train_y[(train_y.shape[0] // 2):].reset_index(drop=True))
        # data finalprocess

        train_x, train_y = self.datafinalprocess(train_y_one_hot,
                                                 train_x,
                                                 train_y[:])

        for count in iterate_count:
            self.train_x.append(train_x[:])
            self.train_y.append(train_y[:])
        for count in iterate_count:
            self.test_x.append(test_x)
        # train & test data 
        print("train & test data....")
        print(type(train_x))
        print(type(train_y))
        train_x = np.array(self.train_x)
        train_y = np.array(self.train_y)
        test_x = np.array(self.test_x)

            # 成交筆數 成交金額 成交股數
        # datavalue for one standard
        # gray_picture
        train_x_list = []
        train_y_list = []
        test_x_list = []

        # flatten the many Dimention
        print("flatten many Dimention")
        for xs, ys in zip(train_x, train_y):
            for x, y in zip(xs, ys):
                train_x_list.append(x)
                train_y_list.append(y)
        for xs in test_x:
            for x in xs:
                test_x_list.append(x)

        train_x = np.array(train_x_list)
        train_y = np.array(train_y_list)
        test_x = np.array(test_x_list)

        one = self.one
        label = self.label
        # print(self.train_x_final,self.train_y_final)
        # self.gray_picture(train_x,train_y)
        test_x = test_x.reshape(-1, train_x.shape[1],train_x.shape[2], 1)

        data = [train_x,  train_y, test_x, one, label,self.df]
        return data

    @staticmethod
    def add_ma_df(df):
        for i in range(5, len(df)):
            df.loc[i, "5ma"] = df["收盤價"][i - 5:i].astype("float").mean()
        df.loc[:, "5ma"] = df["5ma"].fillna(0)
        for i in range(10, len(df)):
            df.loc[i, "10ma"] = df["收盤價"][i - 10:i].astype("float").mean()
        df.loc[:, "10ma"] = df["10ma"].fillna(0)
        # for i in range(20, len(df)):
        #     df.loc[i, "20ma"] = df["收盤價"][i - 20:i].astype("float").mean()
        # df.loc[:, "20ma"] = df["20ma"].fillna(0)

        #  df = df.reindex(columns=['日期', '殖利率(%)', '本益比', '股價淨值比', '成交股數', '成交金額', '開盤價', '最高價', '最低價',
        # '收盤價',  '漲跌價差', '成交筆數'])#'10ma', '20ma',
        return df

    

    def datafinalprocess(self, train_y, train_x, index_y):
        print("data final process ")
        # let the 0 1 -1 have equal mount for  training have equal mount
        day = 20
        print(train_x.shape,"****************")
        train_y = train_y[day:]
        index_y = index_y[day:]

        index_y_buy = index_y[index_y["train_y"] == 1].index.tolist()
        index_y_sell = index_y[index_y["train_y"] == -1].index.tolist()
        index_y_equal = index_y[index_y["train_y"] == 0].index.tolist()
        equal_x = []
        buy_x = []
        sell_x = []
        train_y_test = []
        if len(index_y_buy) > len(index_y_sell):
            sample_mount = len(index_y_sell)
        else:
            sample_mount = len(index_y_buy)

        [buy_x.append(train_x[index - day]) for index in index_y_equal]#[:-3]
        [sell_x.append(train_x[index - day]) for index in index_y_equal]
        [equal_x.append(train_x[index - day]) for index in index_y_equal]
        buy_x = np.array(buy_x[:sample_mount])
        sell_x = np.array(sell_x[:sample_mount])
        equal_x = np.array(equal_x[:sample_mount])
        train_x_test = np.concatenate((buy_x, sell_x))
        train_x = np.concatenate((train_x_test, equal_x))
        [train_y_test.append(train_y[i - day]) for i in index_y_buy]#
        [train_y_test.append(train_y[i - day]) for i in index_y_sell]
        [train_y_test.append(train_y[i - day]) for i in sample(index_y_equal, len(buy_x))]
        train_y = np.array(train_y_test)

        # thinking about how to merge the x data than thing merge y data
        # make the data mount equal
        # train_x shape(-1,days,data mount,1)

        train_x = train_x.reshape(-1,train_x.shape[1],train_x.shape[2] ,1)
        # train_x, _, train_y, _ = tts(train_x, train_y, shuffle=True)
        # train_x, test_x, train_y, test_y = train_x[:int(len(train_x)*0.7)],train_x[int(len(train_x)*0.7):],train_y[:int(len(train_y)*0.7)],train_y[int(len(train_y)*0.7):]
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
        mount = df["成交筆數"].to_numpy().astype(float)
        train_y = np.zeros(num.shape, dtype=int)
        hstock = 0
        day =3
        # could estimate performance and optimise algorithm -> structure first
        for today in range(0, num.shape[0] - day):
            # current idea is if today isn't the extremum  day will not update the value
            # -1 present sale 1 present buy
            monthmax = np.max(num[today+1:day + today])
            monthmin = np.min(num[today+1:day + today])
            monthmean = np.mean(num[today + 1:day + today])

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
