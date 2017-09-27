import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pygeoip
from datetime import date




# path to data used in pygeoip
GEO_DATA_PATH = "/source_data/geoip/GeoLiteCity.dat"
_data_base_path = '/Users/yangyang/Documents/Study/Insight/project/code'
gi = pygeoip.GeoIP(_data_base_path+GEO_DATA_PATH, pygeoip.MEMORY_CACHE)

_dbname = 'mylocaldb' 
_username = 'yangyang'

_con = None
_con = psycopg2.connect(database = _dbname, user = _username)

_sql_query = """
SELECT u.id, u.created_at, u.sign_in_count, u.email, (u.gender is not null) as provide_gender,
u.city, (u.city is not null) as provide_address, u.date_of_birth,
u.last_sign_in_ip,
(u.phone is not null) as provide_phone, (u.account_type='facebook') as registered_by_fb,
w.wish_items,
r.order_number, r.total_expenditures, r.first_order_date
FROM users as u
LEFT JOIN
(
SELECT user_id, COUNT(item_id) as wish_items
FROM item_wishlists
GROUP BY user_id
) as w
ON u.id=w.user_id
LEFT JOIN
(
SELECT renter_id, COUNT(total_cost) as order_number, SUM(total_cost) as total_expenditures,
MIN(created_at) as first_order_date
FROM rentals
GROUP BY 
renter_id
) as r
on r.renter_id = u.id

"""

def load_renter_data():
    """
    Return the df for renter classifier
    """
    return pd.read_sql_query(_sql_query, _con)


####################### Categorical features #######################

def email_category(x):
    """
    Categorize the email accounts.
    """
    MAP = {'msn':'msn',
            'yahoo':'yahoo',
            'gmail':'gmail',
            'hotmail':'hotmail',
            'live.com':'hotmail',
            '.edu':'education',
            'comcast':'comcast_aol_att',
            'aol.com':'comcast_aol_att',
            'att.net':'comcast_aol_att',
            'verizon.net':'comcast_aol_att',
            'icloud':'apple',
            'me.com':'apple',
            'mac.com':'apple',
            'qq':'qq',
            'outlook':'outlook'
            }
    for key in MAP.keys():
        if key in x:
            return MAP[key]
    return "other_email_account"


def city_category(x):
    # cities with more than 50 registered users.
    cities = [
            'New York','San Francisco','Newark',
            'Los Angeles','San Jose','Chicago',
            'Dallas','Atlanta','Ashburn',
            'Miami','London','Seattle',
            'Brooklyn','Boston','Oakland',
            'New Delhi','Toronto','Berkeley',
            'Denver','Singapore','Sydney',
            'Tampa','Melbourne','Houston',
            'San Diego','Anaheim','Mumbai',
            'Seoul', 'Hong Kong','Paris'
            ]
    try:
        for city in cities:
            if city in x:
                return city
        return "Other City"
    except:
        return 'No address'


def transfer_ip_to_city(ip_data):
    city = ip_data.apply(ip_to_city)

    # top 20 biggest cities.
    major_cities = city.value_counts().sort_values(ascending=False).index[0:30].values

    # only keep the biggest cities.
    city = city.apply(lambda x: x if x in major_cities else "Other_cities")
    return city


def ip_to_city(ip):
    try:
        return gi.record_by_addr(ip)['city']
    except:
        return "unknown_by_ip"



def age_category(x):
    # get the age range
    today = date.today()
    try:
        age = today.year - x.year
        if age<25:
            return "under 25"
        elif 25<=age<30:
            return "age25-30"
        elif 30<=age<35:
            return "age30-35"
        elif 35<=age<40:
            return "age35-40"
        else:
            return "40+"
    except:
        return "No_age"
    



















    
