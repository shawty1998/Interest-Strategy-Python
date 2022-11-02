from binance import Client, BinanceSocketManager
import requests
import requests
import pandas as pd
import numpy as np
import time
#from ftx import FtxClient
import time
import urllib.parse
from typing import Optional, Dict, Any, List
import hmac

class FtxClient:
    
    #initially I tried using the ftx api client but that ke[t bugging out so I just took the code I needed and created a class
    
    _ENDPOINT = 'https://ftx.com/api/'

    def __init__(self, api_key=None, api_secret=None, subaccount_name=None) -> None:
        self._session = requests.Session()
        self._api_key = api_key
        self._api_secret = api_secret
        self._subaccount_name = subaccount_name
        
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('GET', path, params=params)
    
    def _post(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('POST', path, json=params)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        request = requests.Request(method, self._ENDPOINT + path, **kwargs)
        self._sign_request(request)
        response = self._session.send(request.prepare())
        return self._process_response(response)

    def _sign_request(self, request: requests.Request) -> None:
        ts = int(time.time() * 1000)
        prepared = request.prepare()
        signature_payload = f'{ts}{prepared.method}{prepared.path_url}'.encode()
        if prepared.body:
            signature_payload += prepared.body
        signature = hmac.new(self._api_secret.encode(), signature_payload, 'sha256').hexdigest()
        request.headers['FTX-KEY'] = self._api_key
        request.headers['FTX-SIGN'] = signature
        request.headers['FTX-TS'] = str(ts)
        if self._subaccount_name:
            request.headers['FTX-SUBACCOUNT'] = urllib.parse.quote(self._subaccount_name)

    def _process_response(self, response: requests.Response) -> Any:
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        else:
            if not data['success']:
                raise Exception(data['error'])
            return data['result']
        
    def place_order(self, market: str, side: str, price: float, size: float, client_id: str,
                    type: str = 'limit', reduce_only: bool = False, ioc: bool = False, post_only: bool = False,
                    ) -> dict:
        return self._post('orders', {'market': market,
                                     'side': side,
                                     'price': price,
                                     'size': size,
                                     'type': type,
                                     'reduceOnly': reduce_only,
                                     'ioc': ioc,
                                     'postOnly': post_only,
                                     'clientId': client_id,
                                     })
    
    def get_open_orders(self, order_id: int, market: str = None) -> List[dict]:
        return self._get(f'orders', {'market': market, 'order_id':order_id})
    
    def get_borrow_rates(self) -> List[dict]:
        return self._get('spot_margin/borrow_rates')
    
    
    


def setup():
    #These are empty accounts, really should store in a seperate encrypted file
    #Bad pracctise especially when sending to others
    
    binance_api_keys = {
    'api_key':'HFTPZvR4Oq0zIj0y0ssmheSKNxaCJDb49UdrSDGm2sSlXMQLRPmhf5eGLOvXvvyh',
    'api_secret':'2scI8hR0zevv17Nj4nP1fECEYlft9eqktxJnrSDhaprMKFYtDGdRG7GfKBShrPKf'
    }

    ftx_api_keys = {
        'api_key':'FTe1CuVsXjvH4m-EF2BFGxH5s4Za-qazfSZL7eA0',
        'api_secret':'9e747Pmd-Z-0PG2XQ8r1AeLKZvd0n5hB6xe-OXhz'
    }
    global bin_client, ftx_client
    
    bin_client = Client(
    binance_api_keys['api_key'],
    binance_api_keys['api_secret'])

    ftx_client = FtxClient(
        ftx_api_keys['api_key'],
        ftx_api_keys['api_secret'])
    
    
def fee_tiers( trading_vol_ftx : int, trading_vol_bin : int, taker=True, bnb_in_binance=0, pay_in_bnb=False):
    
    #takes trading volume and other paramaters and retuens the fee tier that each exchange will put you into
    
    
    """For the sake of this project I am going to neglect the """
    
    order = "Taker" if taker else "Maker"
    
    bin_fees=pd.read_csv(r".\fees\binancefees.csv" ,index_col=False)
    ftx_fees=pd.read_csv(r".\fees\ftxfees.csv" ,index_col=False)
    
    
    try:
        ftx_ = ftx_fees[ftx_fees["Volume"] - trading_vol_ftx <= 0].iloc[-1][order]
    except IndexError:
        ftx_ = ftx_fees.iloc[-1][order]
    
    
    bin_fees = bin_fees[bin_fees["BNB"] - bnb_in_binance <= 0]
    try:
        bin_ = bin_fees[bin_fees["Volume"] - trading_vol_bin <= 0].iloc[-1][order if not pay_in_bnb else order+" BNB"]
    except IndexError:
        bin_ = bin_fees.iloc[-1][order]
    
    return {"ftx" : ftx_,
            "bin" : bin_}



def orderbook_builder( coin_from: str, coin_to: str, ftx_fee: float, bin_fee: float):
    
    #Creating all possible orderbooks from a mixture of the 2 exchanges
    
    order_books=[]
    
    bin_coin_1_coin_2_exist=True
    bin_coin_2_coin_1_exist=True
    
    ftx_orderbook_coin_1_coin_2 = requests.get(r'https://ftx.com/api/markets/'+coin_to+"/"+coin_from+"/orderbook").json()
    ftx_orderbook_coin_2_coin_1 = requests.get(r'https://ftx.com/api/markets/'+coin_from  +"/"+coin_to+"/orderbook").json()
    
    ftx_coin_1_coin_2_exist = ftx_orderbook_coin_1_coin_2['success']
    ftx_coin_2_coin_1_exist = ftx_orderbook_coin_2_coin_1['success']
    
    if ftx_coin_1_coin_2_exist:
        
        ftx_orderbook_coin_1_coin_2=pd.DataFrame(ftx_orderbook_coin_1_coin_2["result"]["asks"],columns=["price","size"])
        ftx_orderbook_coin_1_coin_2["order"]="ask"
        ftx_orderbook_coin_1_coin_2["exchange"]="ftx"
        ftx_orderbook_coin_1_coin_2["priceWithFee"]=ftx_orderbook_coin_1_coin_2["price"]*(1+ftx_fee)
        
    if ftx_coin_2_coin_1_exist:
        
        ftx_orderbook_coin_2_coin_1=pd.DataFrame(ftx_orderbook_coin_2_coin_1["result"]["bids"],columns=["price","size"])
        
        ftx_orderbook_coin_2_coin_1["size"]=ftx_orderbook_coin_2_coin_1["price"]*ftx_orderbook_coin_2_coin_1["size"]
        ftx_orderbook_coin_2_coin_1["price"]=1/ftx_orderbook_coin_2_coin_1["price"]
        ftx_orderbook_coin_2_coin_1["order"]="bid"
        ftx_orderbook_coin_2_coin_1["exchange"]="ftx"
        ftx_orderbook_coin_2_coin_1["priceWithFee"]=ftx_orderbook_coin_2_coin_1["price"]*(1+ftx_fee)

    
    try:
        bin_orderbook_coin_1_coin_2 = bin_client.get_order_book(symbol = coin_to + coin_from )
        
        bin_orderbook_coin_1_coin_2=pd.DataFrame(bin_orderbook_coin_1_coin_2["asks"],columns=["price","size"])
        bin_orderbook_coin_1_coin_2["price"]=bin_orderbook_coin_1_coin_2["price"].astype(float)
        bin_orderbook_coin_1_coin_2["size"]=bin_orderbook_coin_1_coin_2["size"].astype(float)
        bin_orderbook_coin_1_coin_2["order"]="ask"
        bin_orderbook_coin_1_coin_2["exchange"]="bin"
        bin_orderbook_coin_1_coin_2["priceWithFee"]=bin_orderbook_coin_1_coin_2["price"]*(1+bin_fee)
        
        if ftx_coin_1_coin_2_exist:

            order_books.append(comb_orderbook( ftx_orderbook_coin_1_coin_2, bin_orderbook_coin_1_coin_2))
        
        if ftx_coin_2_coin_1_exist:
            order_books.append(comb_orderbook( ftx_orderbook_coin_2_coin_1, bin_orderbook_coin_1_coin_2))
        
        if (not ftx_coin_1_coin_2_exist) and (not ftx_coin_2_coin_1_exist):
            order_books.append( orderbook_formating( bin_orderbook_coin_1_coin_2))
            
    except:
        bin_coin_1_coin_2_exist=False
    
    try:
        time.sleep(1)
        bin_orderbook_coin_2_coin_1 = bin_client.get_order_book(symbol = coin_from + coin_to )
                                                          
        bin_orderbook_coin_2_coin_1=pd.DataFrame(bin_orderbook_coin_2_coin_1["bids"],columns=["price","size"])
        
        bin_orderbook_coin_2_coin_1["size"]=(bin_orderbook_coin_2_coin_1["price"].astype(float)*
                                             bin_orderbook_coin_2_coin_1["size"].astype(float))
        bin_orderbook_coin_2_coin_1["price"]=1/(bin_orderbook_coin_2_coin_1["price"].astype(float))
        bin_orderbook_coin_2_coin_1["order"]="bid"
        bin_orderbook_coin_2_coin_1["exchange"]="bin"
        bin_orderbook_coin_2_coin_1["priceWithFee"]=bin_orderbook_coin_2_coin_1["price"]*(1+bin_fee)
        

        if ftx_coin_1_coin_2_exist:
            order_books.append(comb_orderbook( ftx_orderbook_coin_1_coin_2, bin_orderbook_coin_2_coin_1))
        
        if ftx_coin_2_coin_1_exist:
            order_books.append(comb_orderbook( ftx_orderbook_coin_2_coin_1, bin_orderbook_coin_2_coin_1))
            
        if (not ftx_coin_1_coin_2_exist) and (not ftx_coin_2_coin_1_exist):
            order_books.append( orderbook_formating( bin_orderbook_coin_2_coin_1))
                                                    
    except:
        bin_coin_2_coin_1_exist=False
    
    
    if (not bin_coin_1_coin_2_exist) and (not bin_coin_2_coin_1_exist):
        if ftx_coin_1_coin_2_exist:
                order_books.append( orderbook_formating( ftx_orderbook_coin_1_coin_2))
                
        if ftx_coin_2_coin_1_exist:
                order_books.append( orderbook_formating( ftx_orderbook_coin_2_coin_1))
    
    return order_books


def comb_orderbook( ftx_orderbook, bin_orderbook):
    #combining the orderbooks
    joined_orderbook=pd.concat([ftx_orderbook, bin_orderbook])
    joined_orderbook=joined_orderbook.sort_values(by=["priceWithFee"], ascending=True)
    joined_orderbook["cumSum"]=joined_orderbook["size"].cumsum()
    joined_orderbook=joined_orderbook.reset_index(drop=True)
    return joined_orderbook

def orderbook_formating( orderbook):
    #formating orderbooks when we don't need to combine
    orderbook=orderbook.sort_values(by=["priceWithFee"], ascending=True)
    orderbook["cumSum"]=orderbook["size"].cumsum()
    orderbook=orderbook.reset_index(drop=True)
    return orderbook   


def best_execution( quantity, order_books):
    
    #Return the cost of executing on each mixture of exchanges
    
    book_price_pairs=[]
    for book in order_books:
        max_size=book.iloc[-1]["cumSum"]
        if max_size < quantity:
            book_price_pairs.append((book,sum(book["size"]*book["priceWithFee"]), max_size))
            continue
        book_len=len(book[book["cumSum"]<quantity])
        book.loc[book_len,"size"] = book.loc[book_len,"size"] - (book.loc[book_len ,"cumSum"]-quantity)
        book=book.iloc[:book_len+1]
        book_price_pairs.append((book,sum(book["size"]*book["priceWithFee"]), quantity))
    return book_price_pairs



    
    
def execute(bid_orderbook, ask_orderbook, coin_from, coin_to):
    
    #Sending orders from the execution question to each exchange 

    #buyer in coin_tocoin_from seller in coin_fromcoin_to
    #Opposisate to before if there is an ask we want to lift and hit the bids
    
    bin_buys=ask_orderbook[ask_orderbook["exchange"]=="bin"]
    bin_sells=bid_orderbook[bid_orderbook["exchange"]=="bin"]
    
    ftx_buys=ask_orderbook[ask_orderbook["exchange"]=="ftx"]
    ftx_sells=bid_orderbook[bid_orderbook["exchange"]=="ftx"]
    
    
    #TODO add in a catch mechnism that receives the info back from the exchange to see what actually getsexecuted
    #Python is quite slow so very unlikley that the OB hasn't changed before we make trades
    #This is why we use IOC orders, should really have the print out after receiving confirmation from the exchange
    """
    if not bin_buys.empty:
        for i,row in bin_buys.iterrows():
            bin_client.create_test_order(
            symbol=coin_to+coin_from,
            side=bin_client.SIDE_BUY,
            type=bin_client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_IOC,
            quantity=float(row["size"]),
            price=str(row["price"]))
            
    if not bin_sells.empty:
        for i, row in bin_sells.iterrows():
            bin_client.create_test_order(
            symbol=coin_from+coin_to,
            side=bin_client.SIDE_SELL,
            type=bin_client.ORDER_TYPE_LIMIT,
            timeInForce=bin_client.TIME_IN_FORCE_IOC,
            quantity=float(row["size"]),
            price=str(row["price"]))
            
    if not ftx_buys.empty:
        for i, row in ftx_buys.iterrows():  
            ftx_client.palce_order(
            market = coin_to+"/"+coin_from,
            side = "buy",
            price = row["price"],
            type = "limit",
            size = row["size"],
            ioc = True )

    if not ftx_sells.empty:
        for i, row in ftx_sells.iterrows():  
            ftx_client.palce_order(
            market = coin_to+"/"+coin_from,
            side = "sell",
            price = row["price"],
            type = "limit",
            size = row["size"],
            ioc = True )
    """
    
    
def to_execute_main( trading_vol_ftx, trading_vol_bin ,coin_from, coin_to, size):
    
    #The main function for the execution question
    
    setup()
    ftx_fee, bin_fee=fee_tiers( trading_vol_ftx, trading_vol_bin, taker=True, bnb_in_binance=0, pay_in_bnb=False).values()
    order_books = orderbook_builder( coin_from, coin_to, ftx_fee, bin_fee)
    
    output_list=best_execution( size, order_books)

    best_orderbook=output_list[np.argmin([output[1]/output[2] for output in output_list])]



    print(f"Able to execute {best_orderbook[2]} lots at a total price of {round(best_orderbook[1],2)} {coin_from}, for a price of\
 {round(best_orderbook[1]/best_orderbook[2],2)} {coin_from} per {coin_to}")

    orderbook=best_orderbook[0]
    
    #resplitting the orderbooks so that we have bids and asks seperate
    bid_orderbook=orderbook[orderbook["order"]=="bid"]
    ask_orderbook=orderbook[orderbook["order"]=="ask"]
    
    #converting implied asks back into bids should still be sorted by importance
    bid_orderbook["size"]=bid_orderbook["price"]*bid_orderbook["size"]
    bid_orderbook["price"]=1/(bid_orderbook["price"])

    execute(bid_orderbook[["exchange","size","price"]],
            ask_orderbook[["exchange","size","price"]], coin_from, coin_to)
    
    
def bids_orderbook_builder( market: str, ftx_fee: float, bin_fee: float):
    

    #Creates all bid side order books that we can hit in order to short both spot and futures
    
    order_books={}
    
    
    ftx_spot_orderbook = requests.get(r'https://ftx.com/api/markets/'+market+"/USD/orderbook").json()
    ftx_future_orderbook = requests.get(r'https://ftx.com/api/markets/'+market+"-PERP"+"/orderbook").json()
    
    if ftx_spot_orderbook['success']:
        
        ftx_spot_orderbook=pd.DataFrame(ftx_spot_orderbook["result"]["bids"],columns=["price","size"])
        ftx_spot_orderbook["priceWithFee"]=ftx_spot_orderbook["price"]*(1-ftx_fee)
        ftx_spot_orderbook["exchange"]="ftx"

        order_books["ftx_spot"]=orderbook_formating( ftx_spot_orderbook).sort_values(
                by=["priceWithFee"], ascending=False)[["price","size","priceWithFee","exchange"]].reset_index(drop=True)
        
    
    if ftx_future_orderbook['success']:
        
        ftx_future_orderbook=pd.DataFrame(ftx_future_orderbook["result"]["bids"],columns=["price","size"])
        ftx_future_orderbook["priceWithFee"]=ftx_future_orderbook["price"]*(1-ftx_fee)
        ftx_future_orderbook["exchange"]="ftx"

        order_books["ftx_future"]=orderbook_formating( ftx_future_orderbook).sort_values(
                by=["priceWithFee"], ascending=False)[["price","size","priceWithFee","exchange"]].reset_index(drop=True)
        
    
        
    
    try:
        bin_spot_orderbook = bin_client.get_order_book( symbol = market+"BUSD" )
        bin_spot_orderbook=pd.DataFrame(bin_spot_orderbook["bids"],columns=["price","size"])
        bin_spot_orderbook["price"]=bin_spot_orderbook["price"].astype(float)
        bin_spot_orderbook["size"]=bin_spot_orderbook["size"].astype(float)
        bin_spot_orderbook["priceWithFee"]=bin_spot_orderbook["price"]*(1-bin_fee)
        bin_spot_orderbook["exchange"]="bin"

        order_books["bin_spot"]=orderbook_formating(bin_spot_orderbook).sort_values(
                by=["priceWithFee"], ascending=False)[["price","size","priceWithFee","exchange"]].reset_index(drop=True)

        """It would be be more correct especially in a production environment to rewrite this function with of that
        returns a dataframe with the entries inverted but I already have so may functions
        """
    
            
    except:
        print("No Binance Spot OrderBook available for Symbol")
        
    
    try:
        time.sleep(1)
        bin_futures_orderbook = bin_client.futures_order_book(symbol = market+"BUSD" )

        bin_futures_orderbook=pd.DataFrame(bin_futures_orderbook["bids"],columns=["price","size"])
        bin_futures_orderbook["price"]=bin_futures_orderbook["price"].astype(float)
        bin_futures_orderbook["size"]=bin_futures_orderbook["size"].astype(float)
        bin_futures_orderbook["priceWithFee"]=bin_futures_orderbook["price"]*(1-bin_fee)
        bin_futures_orderbook["exchange"]="bin"
        

        

        order_books["bin_future"]=orderbook_formating(bin_futures_orderbook).sort_values(
            by=["priceWithFee"], ascending=False)[["price","size","priceWithFee","exchange"]].reset_index(drop=True)
            
    except:
        print("No Binance Futures OrderBook available for Symbol")
    
    
    return order_books


def asks_orderbook_builder( current: str, market: str, ftx_fee: float, bin_fee: float):
    
        #takes what we are currently holding our positions in so we can pair them off against bids in the opposing book


        if current == "ftx_spot":

            ftx_spot_orderbook = requests.get(r'https://ftx.com/api/markets/'+market+"/USD/orderbook").json()

            ftx_spot_orderbook=pd.DataFrame(ftx_spot_orderbook["result"]["asks"],columns=["price","size"])
            ftx_spot_orderbook["priceWithFee"]=ftx_spot_orderbook["price"]*(1+ftx_fee)
            ftx_spot_orderbook["exchange"]="ftx"

            return orderbook_formating( ftx_spot_orderbook)[["price","size","priceWithFee","exchange"]]


        elif current == "ftx_future":

            ftx_future_orderbook = requests.get(r'https://ftx.com/api/markets/'+market+"-PERP"+"/orderbook").json()

            ftx_future_orderbook=pd.DataFrame(ftx_future_orderbook["result"]["asks"],columns=["price","size"])
            ftx_future_orderbook["priceWithFee"]=ftx_future_orderbook["price"]*(1+ftx_fee)
            ftx_future_orderbook["exchange"]="ftx"

            return orderbook_formating( ftx_future_orderbook)[["price","size","priceWithFee","exchange"]]

        elif current == "bin_spot":

            bin_spot_orderbook = bin_client.get_order_book( symbol = market+"BUSD" )
            bin_spot_orderbook=pd.DataFrame(bin_spot_orderbook["bids"],columns=["price","size"])
            bin_spot_orderbook["price"]=bin_spot_orderbook["price"].astype(float)
            bin_spot_orderbook["size"]=bin_spot_orderbook["size"].astype(float)
            bin_spot_orderbook["priceWithFee"]=bin_spot_orderbook["price"]*(1+bin_fee)
            bin_spot_orderbook["exchange"]="bin"

            return orderbook_formating(bin_spot_orderbook)[["price","size","priceWithFee","exchange"]]



        elif current == "bin_future":
            bin_futures_orderbook = bin_client.futures_order_book(symbol = market+"BUSD" )

            bin_futures_orderbook=pd.DataFrame(bin_futures_orderbook["asks"],columns=["price","size"])
            bin_futures_orderbook["price"]=bin_futures_orderbook["price"].astype(float)
            bin_futures_orderbook["size"]=bin_futures_orderbook["size"].astype(float)
            bin_futures_orderbook["priceWithFee"]=bin_futures_orderbook["price"]*(1+bin_fee)
            bin_futures_orderbook["exchange"]="bin"

            return orderbook_formating(bin_futures_orderbook)[["price","size","priceWithFee","exchange"]]

        
    

        
def asks_ob_retry( current, market , ftx_fee, bin_fee):
    
    #retries the orderbook query above as is ofter rejected by exchange
    
    tries=0
    while tries <3:
        try:
            return asks_orderbook_builder( current, market, ftx_fee, bin_fee)
        except:

            tries+=1
            time.sleep(1)
            
            
def bids_ob_retry(market , ftx_fee, bin_fee):
    
    #retries the orderbook query above as is ofter rejected by exchange
    
    tries=0
    while tries <3:
        try:
            return bids_orderbook_builder( market, ftx_fee, bin_fee)
        except:

            tries+=1
            time.sleep(1)   



async def bin_futures_info(coin):
    #pulls the mark price, index price and current binance funding rate
    
    bm = BinanceSocketManager(bin_client, user_timeout=60)

    ts = bm.symbol_mark_price_socket(symbol= coin+"BUSD")

    await ts.__aenter__()

    msg = await ts.recv()

    await ts.__aexit__(None, None, None)
    
    """
    "e": "markPriceUpdate",     // Event type
    "E": 1562305380000,         // Event time
    "s": "BTCUSDT",             // Symbol
    "p": "11794.15000000",      // Mark price
    "i": "11784.62659091",      // Index price
    "P": "11784.25641265",      // Estimated Settle Price, only useful in the last hour before the settlement starts
    "r": "0.00038167",          // Funding rate
    "T": 1562306400000          // Next funding time
    """
    return {"futures_price":msg["data"]["i"],
            "coin_price":msg["data"]["p"],
            "funding_rate":msg["data"]["r"]}



#create interst rate dict for margin question
async def bin_interest_attributes(coin):
    #takes the futures data and adds the margin interest rate

    futures_data = await bin_futures_info(coin)
    
    futures_data["funding_rate"] = 3*float(futures_data["funding_rate"])

    #binance interest are calculated on an hourly basis
    tries=0
    while tries < 3:
        try:
            margin = bin_margin(coin)
            futures_data["margin_interest"] = margin
            return {
        "bin_spot":(futures_data["coin_price"], futures_data["margin_interest"]),
        "bin_future":(futures_data["futures_price"], futures_data["funding_rate"])
            }
            
        except:
            tries += 1
            
    return {
        "bin_future":(futures_data["futures_price"], futures_data["funding_rate"])
    }
            
    
    

def bin_margin(coin):
    #Get the cross margin daily interest from binance
    return bin_client.get_cross_margin_data(coin="BTC")[0]["dailyInterest"]


def ftx_interest_attributes(coin):
    
    #getting all the interet attributes needed for the ftx exchange
    
    funding=requests.get(f'https://ftx.com/api/futures/{coin}-PERP/stats').json()["result"]["nextFundingRate"]

    ftx_prices=requests.get(f'https://ftx.com/api/futures/{coin}-PERP').json()["result"]


    funding_rate_daily = 24*float(funding)
    #Ftx Funding rates are hourly and margin rates are also hourly so need to convert to daily to compare
    #
    #This endpoint is very flimsy and will reject calls alot of the time
    tries = 1
    
    while tries < 3:
        try:
            all_borrow_ftx=pd.DataFrame(ftx_client.get_borrow_rates())
            tries=3
        except:
            time.sleep(.25)
            tries += 1
    
    
    #This is to try and deal with the issue of ftx not liking being pinged too much
    
    
    try:
        margin_interest=float(all_borrow_ftx[all_borrow_ftx["coin"]==coin]["estimate"])
    except TypeError:
        return {"ftx_future":(ftx_prices.get('index'),funding_rate_daily)}
    
    
    #TODO Add some bits to just return margin pieces also have to do the same for the binance function


    margin_interest_daily = 24*float(margin_interest)
    
    return  {"ftx_spot":(ftx_prices.get('mark'), margin_interest_daily),
            "ftx_future":(ftx_prices.get('index'),funding_rate_daily)}
    
    
def funding_difference( previous_rate: float , new_rate: float, old_mark: float, new_mark: float, time_days: float):
    #calculating the difference between the rate we are currently paying and what we will be if we change
    return old_mark*(1+previous_rate*time_days)-new_mark*(1+new_rate*time_days)


       
            
            
async def optimal_interest(coin : str, current: str, time_days: float, mark : float):
    
    #calculated the interest from going from the current position to all other and returns all differences when positive

    bin_interest=await bin_interest_attributes(coin)
    ftx_interest=ftx_interest_attributes(coin)

    interest = {**bin_interest, **ftx_interest}

    current_interest=interest.pop(current)
    #This is  not the real mark, it would have been set when the trade was initially put on
    #need to change
    if mark==0:
        mark=float(current_interest[0])

    interest_df=pd.DataFrame({"from_instrument":[current for i in range(3)],
                 "to_instrument":[instrument for instrument in interest.keys()],
                  "from_insterest":[current_interest[1] for i in range(3)],
                  "to_insterest":[float(value[1]) for value in interest.values()],
                  "from_mark":[mark for i in range(3)],
                  "to_mark":[value[0] for value in interest.values()]
                 })


    interest_df["improvment"]=interest_df.apply( lambda row : funding_difference(previous_rate =  float(row["from_insterest"]),
                                                                                new_rate = float(row["to_insterest"]),
                                                                                old_mark = float(row["from_mark"]),
                                                                                new_mark = float(row["to_mark"]),
                                                                                time_days = time_days),axis=1)


    possible_trades=interest_df[interest_df["improvment"]>0]
    possible_trades=possible_trades[["from_instrument","to_instrument","improvment"]].sort_values(by=["improvment"]
                                                                                                  , ascending=False).reset_index(drop=True)
    
    return possible_trades



def interest_execute(coin, current, to, size, price_bid, price_ask):
    
    #executes the interest changing code on exchange borrowing and repaying loans on binence
    #But FTX handles all that for you as long as you are holding colateral
    
    """
    if current = "bin_future":
        binance_client.futures_create_order(
        symbol= coin + "BUSD" ,
        side=bin_client.SIDE_BUY,
        type=bin_client.ORDER_TYPE_LIMIT,
        timeInForce=Client.TIME_IN_FORCE_IOC,
        quantity=size,
        price=str(price_ask))
    
    elif current = "ftx_future":
        ftx_client.palce_order(
        market = coin + "-PERP",
        side = "buy",
        price = price_ask,
        type = "limit",
        size = size,
        ioc = True )
    
    elif current = "bin_spot":
        bin_client.create_test_order(
        symbol= coin + "BUSD" ,
        side=bin_client.SIDE_BUY,
        type=bin_client.ORDER_TYPE_LIMIT,
        timeInForce=Client.TIME_IN_FORCE_IOC,
        quantity=size,
        price=str(price_ask))
        
        #Repay the Binance loan
        bin_client.repay_margin_loan(asset=coin, amount= str(size))
        
    elif current = "ftx_spot":
        ftx_client.palce_order(
        market = coin + "USD",
        side = "buy",
        price = price_ask,
        type = "limit",
        size = size,
        ioc = True )
    
    if to = "bin_future":
        binance_client.futures_create_order(
        symbol= coin + "BUSD" ,
        side=bin_client.SIDE_SELL,
        type=bin_client.ORDER_TYPE_LIMIT,
        timeInForce=Client.TIME_IN_FORCE_IOC,
        quantity=size,
        price=str(price_bid))
    
    elif to  = "ftx_future":
        ftx_client.palce_order(
        market = coin + "-PERP",
        side = "sell",
        price = price_bid,
        type = "limit",
        size = size,
        ioc = True )
    
    
    elif to  = "bin_spot":
        
        #have to borrow to short
        bin_client.create_margin_loan(asset=coin, amount=str(size))
        
        bin_client.create_margin_order((
        symbol= coin + "BUSD" ,
        side=bin_client.SIDE_SELL,
        type=bin_client.ORDER_TYPE_LIMIT,
        timeInForce=Client.TIME_IN_FORCE_IOC,
        quantity=size,
        price=str(price_bid))
    
    elif to  = "ftx_spot":
    
        #You don't have to specify margin on ftx it does it for you
        
        ftx_client.palce_order(
        market = coin + "USD",
        side = "sell",
        price = price_bid,
        type = "limit",
        size = size,
        ioc = True )
        """
    return


def trade_creator_interest( asks, bids, coin : str, ftx_fee : float, bin_fee : float, saving: float, total_size, current : str, to : str):

    #pairs bids and offers and executes them if the their difference is less than the amount saved and send to execution function
        
    size_bid=bids.loc[0,"size"]
    size_ask=asks.loc[0,"size"]

    price_bid=bids.loc[0,"priceWithFee"]
    price_ask=asks.loc[0,"priceWithFee"]
    
    price_bid_exe=bids.loc[0,"price"]
    price_ask_exe=asks.loc[0,"price"]

    trade_size_total=0

    while price_ask-price_bid<saving and trade_size_total < total_size:
        
        
        if size_bid==size_ask:
            bids=bids.drop(0)
            asks=asks.drop(0)
            trade_size = min(size_bid, total_size - trade_size_total )
            #print("equal",size_bid,total_size,trade_size_total,len(bids),len(asks))
            #To code to send to an execting program

        elif size_bid>size_ask:
            asks=asks.drop(0)
            bids.loc[0,"size"]=bids.loc[0,"size"]-size_ask
            trade_size = min(size_ask, total_size - trade_size_total )
            #print("larger_bid",size_ask,total_size,trade_size_total,len(asks))
            #To code to send to an execting program

        elif size_bid<size_ask:
            bids=bids.drop(0)
            asks.loc[0,"size"] = asks.loc[0,"size"]-size_bid
            trade_size = min(size_bid, total_size - trade_size_total )
            #print("larger_ask",size_bid,total_size-trade_size_total, len(bids))
            #To code to send to an execting program
            
        else:
            print("something has gone wrong")
            return
        

            

        print(f"Shorting {trade_size} of {to} at {price_bid} and buying back {trade_size} of {current}\
 at {price_ask} saving {round(trade_size*(saving-(price_ask-price_bid)) ,2)}")
        


        bids=bids.reset_index(drop=True)
        asks=asks.reset_index(drop=True)
        
        interest_execute(coin=coin, current=current, to=to,
                    size=trade_size, price_bid=price_bid_exe, price_ask=price_ask_exe)


        if len(bids)==0 or len(asks)==0:
            break
            


        size_bid=bids.loc[0,"size"]
        size_ask=asks.loc[0,"size"]

        price_bid=bids.loc[0,"priceWithFee"]
        price_ask=asks.loc[0,"priceWithFee"]
        
        price_bid_exe=bids.loc[0,"price"]
        price_ask_exe=asks.loc[0,"price"]

        trade_size_total += trade_size
        
        
    return(asks , total_size-trade_size_total)



async def interest_main(coin, current, total_size, time_days,  trading_vol_ftx, trading_vol_bin, mark=0):
    
    #main function for interest question
    
    setup()
    
    ftx_fee, bin_fee=fee_tiers( trading_vol_ftx, trading_vol_bin, taker=True, bnb_in_binance=0, pay_in_bnb=False).values()
    
    possible_trades=await optimal_interest(coin = coin, current = current, time_days = time_days, mark=mark)
    
    num_trades=len(possible_trades)

    if num_trades==0:
        print("You have the optimal rate, nice!")
        return
    else:
        print(possible_trades)

    asks=asks_ob_retry( current = current, market = coin, ftx_fee = ftx_fee, bin_fee = bin_fee)
    bids=bids_ob_retry( market = coin, ftx_fee = ftx_fee, bin_fee = bin_fee)

    #really the best way to do this would be to use pointers to the original df in memory and just iterate
    #it would be much better if you wanted to add exchanges without having to rewrite

    if num_trades>0 :
        asks, total_size = trade_creator_interest( asks = asks,
                               bids = bids[possible_trades.loc[0,"to_instrument"]],
                               coin = coin,
                               ftx_fee = ftx_fee,
                               bin_fee = bin_fee,
                               saving=possible_trades.loc[0,"improvment"],
                               total_size=total_size,
                               current=current,
                               to=possible_trades.loc[0,"to_instrument"])

    if num_trades>1 and len(asks) != 0 and total_size !=0 :
        asks, total_size = trade_creator_interest( asks = asks,
                               bids = bids[possible_trades.loc[1,"to_instrument"]],
                               coin = coin,
                               ftx_fee = ftx_fee,
                               bin_fee = bin_fee,
                               saving=possible_trades.loc[1,"improvment"],
                               total_size=total_size,
                               current=current,
                               to=possible_trades.loc[1,"to_instrument"])

    if num_trades>2 and len(asks) != 0 and total_size !=0:
        trade_creator_interest( asks = asks,
                               bids = bids[possible_trades.loc[2,"to_instrument"]],
                               coin = coin,
                               ftx_fee = ftx_fee,
                               bin_fee = bin_fee,
                               saving=possible_trades.loc[2,"improvment"],
                               total_size=total_size,
                               current=current,
                               to=possible_trades.loc[1,"to_instrument"])
        
    return