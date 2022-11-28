import threading
import pandas as pd
import pandas_datareader.data as web
import datetime
from sklearn import linear_model
from time import sleep
import joblib

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    OKRED = '\033[40m'
    OKWHITE = '\33[41'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def getprice(label="TATAPOWER.NS", sleep_time=1):

    sleep(sleep_time)
    quote = web.get_quote_yahoo(label)  # Using yahoo finance to get the stock market live data
    price = quote["price"].values[0]
    current_time = datetime.datetime.now()
    return tuple([price, current_time])


# This Is the function that trains the model using the input data(dataframe)
def train(input):
    print("\nPrabhanshu Stock market prediction model is updating...", end=" ")

    featureMat = input.iloc[:, : len(input.columns) - 1]
    label = input[input.columns[-1]]

    model = linear_model.LinearRegression()
    model.fit(featureMat, label)

    joblib.dump(model, "modelLR.pkl")
    print("[Completed]")


number_of_features = 5
training_record_criterian = 5
number_of_predictions = 3

#from here now data set starts

data = pd.DataFrame(columns=range(number_of_features))

predict_input = list()

while 1:

    feature = list()

    for i in range(number_of_features):

        price = getprice()[0]
        feature.append(price)
        predict_input.append(price)

        try:

            first_predict = True
            model = joblib.load("modelLR.pkl")
            print("")
            inputlist = predict_input.copy()

            for feature_value in inputlist[-(3):]:
                print(f"{bcolors.OKBLUE} --> ", int(feature_value * 100) / 100, end=" ")

            price = getprice(sleep_time=0)[0]

            for i in range(number_of_predictions):
                pre_price = model.predict([inputlist[-(number_of_features - 1):]])

                print(f"{bcolors.OKCYAN} --> ", int(pre_price[0] * 100) / 100, end=" ")

                if first_predict:

                    if pre_price[0] - inputlist[-1] > 0:
                        print(f"{bcolors.OKWHITE}  \u2191", end="")

                        print(f"{bcolors.BOLD}[", int((pre_price[0] - price) * 1000000 / price) / 10000, "%] ", end=" ")
                        print(f"{bcolors.OKGREEN} Actual: ", price, end="")

                    elif pre_price[0] - inputlist[-1] == 0:
                        print(f"{bcolors.HEADER} \u2022", end="")
                        print(f"{bcolors.BOLD}[", int((pre_price[0] - price) * 1000000 / price) / 10000, "%] ", end=" ")
                        print(f"{bcolors.OKGREEN} Actual: ", price, end="")

                    else:
                        print(f"{bcolors.FAIL}  \u2193", end="")
                        print(f"{bcolors.BOLD}[", int(-(pre_price[0] - price) * 1000000 / price) / 10000, "%] ",
                              end=" ")
                        print(f"{bcolors.OKGREEN} Actual: ", price, end="")

                    if price - inputlist[-1] > 0:
                        print(f"{bcolors.OKGREEN}  \u2191", end=" ")
                    elif price - inputlist[-1] == 0:
                        print(f"{bcolors.HEADER} \u2022", end="")
                    else:
                        print(f"{bcolors.FAIL}  \u2193", end=" ")

                    first_predict = False

                inputlist.append(pre_price[0])
        except:

            print("Please Wait Prabhanshu Pal Stock Predicter model is getting ready...")

    data.loc[len(data.index)] = feature

    if len(data.index) % training_record_criterian == 0:

        trainer = threading.Thread(target=train, args=(data,))
        trainer.start()
        trainer.join()
