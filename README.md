# DeepPapers2Recommend

This application can be used to search for and get recommendations for papers accepted at various recent top CS conferences and workshops. The conferences represented include ICLR (2017, 2018), ICML (2018), and NIPS (2016, 2017, 2018). The workshops represented include ICLR (2017, 2018), some of ICML (2018), and NIPS (2018). A total of 2707 papers are represented as of this version.

The front-end is implemented using Flask. All recommendation processing is done using Gensim. Data is collected by scraping data from webpages using BeautifulSoup (bs4).

## Dependencies

This application uses Python3.

You will need flask, gensim, scipy, numpy, scikit-learn, and bs4 (if you want to update the database). These can be setup using `pip` manually or using the command below.

```
pip install -r requirements.txt
```

## AWS Deployement

The current version of the application is available [here](http://deeppapers2recommend.us-east-1.elasticbeanstalk.com). The website is hosted using AWS Elastic Beanstalk.

## Steps to Run Application Locally

The repo has a pickle file (`paperdata.pkl`) which can be directly used. If you would like to create or update the database, run the following:
```
python save_raw_data.py
```

Then, to run a command line version of the recommendation tool, run the following:
```
python recommend.py
```

To run the GUI which is created using Flask, run the following in Linux:
```
export FLASK_APP=flaskr
export FLASK_ENV=development //Optional, to enable debug mode
flask run
```

In Windows, run the following:
```
set FLASK_APP=flaskr
set FLASK_ENV=development //Optional, to enable debug mode
flask run
```

Now, navigate to the address/port indicated by flask and append it with `/search`. For example, if the address/port is 127.0.0.1:5000, navigate to `127.0.0.1:5000/search`.

For all search queries, the default maximum papers displayed is 10. This can be modified by adding a `GET` parameter `limit=<your limit>` to the link. For example, assuming the query was `machine learning`, the URL could be `127.0.0.1:5000/search?q=machine+learning&limit=50`.

The default maximum recommendations shown is also 10. Thus, a similar procedure exists for the recommendation page (which would be `127.0.0.1:5000/recommend`), where adding a `limit=<your limit>` `GET` parameter to the URL will adjust the amount of recommendations shown.
