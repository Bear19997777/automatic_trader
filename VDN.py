
def VDN(df,result):
    money = 0
    stock = 0
    last_msg = "NoTransaction"
    global last_y
    # validation control
    for row in range(len(df)):#len(df)-219 438,
        if df[result][row] == 1  and float(df["收盤價"][row])!= 0 and stock==0:
            money_buy = float(df["收盤價"][row])
            print(str(df["date"][row]),money_buy,"買")
            money -= float(df["收盤價"][row])
            stock += 1
            last_msg = str(df["date"][row])+" "+str(money_buy)+" "+"買"
            print(f"lastmsgbuy{last_msg}")
            # if row == (len(df)-1):return last_msg
        elif df[result][row] == -1 and stock==1:
            money +=float(df["收盤價"][row ])
            print(str(df["date"][row]),float(df["收盤價"][row]),"賣")
            stock =0
            last_msg = str(df["date"][row])+" "+str(float(df["收盤價"][row]))+" "+"賣"
            print(f"lastmsgsell{last_msg}")
            # if row == (len(df) - 1): return last_msg
        # elif df[result][row] == 0:
        #
        #     last_msg = str(df["日期"][row]) + " " + str(float(df["收盤價"][row])) + " " + "平"
        #     # if row == (len(df) - 1): return last_msg
    sell_out = float(df["收盤價"].tail(1))*stock
    print(f"result of {result}")
    print("sell_out money",sell_out)
    money = money+sell_out
    print("last money:",money)
    dict = {"last_msg":last_msg,"last_money":money}
    print(dict)
    return dict
# print(money)