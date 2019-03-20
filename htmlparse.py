from bs4 import BeautifulSoup
import requests

#nips2017url = "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-30-2017"
#urlResp = requests.get(nips2017url)
#icml2018url = "http://proceedings.mlr.press/v80/"
#soup = BeautifulSoup(urlResp.text, 'html.parser')

def initDB_nips(infoUrl, fullDB, venue):
    urlResp = requests.get(infoUrl)
    soup = BeautifulSoup(urlResp.text, 'html.parser')
    cnt = 0
    for item in soup.find_all('li'):
        link = item.a.get('href')
        if link == "/":
            continue
        link = "https://papers.nips.cc" + link
        paperResp = requests.get(link)
        paperSoup = BeautifulSoup(paperResp.text, 'html.parser')
        title = paperSoup.title
        abstract = paperSoup.find('p', attrs={'class': 'abstract'})
        authors = paperSoup.find_all('li', attrs={'class': 'author'})
        authorsExt = [item.get_text() for item in authors]
        pdf = paperSoup.find('meta', attrs={'name':'citation_pdf_url'})
        pdfcontent = ""
        if pdf:
            pdfcontent = pdf.get("content")
        info = {'title': title.get_text(),
                'abstract': abstract.get_text(),
                'authors': authorsExt,
                'pdf': pdfcontent,
                'keywords': []}
        fullDB[title.get_text()] = [info, 'Accept', venue]
        cnt+=1
        print(cnt)
        #if cnt >= 50:
        #    break
def initDB_icml(infoUrl, fullDB, venue):
    urlResp = requests.get(infoUrl)
    soup = BeautifulSoup(urlResp.text, 'html.parser')
    cnt = 0
    for item in soup.find_all('p', attrs={'class': 'links'}):
        link = item.a.get('href')
        paperResp = requests.get(link)
        paperSoup = BeautifulSoup(paperResp.text, 'html.parser')
        paperInfo = paperSoup.find('article')
        title = paperInfo.h1
        abstract = paperInfo.find('div', attrs={'class', 'abstract'})
        authors = paperInfo.find('div', attrs={'class', 'authors'})
        authorsExt = authors.get_text().replace('\n', '').replace(';', '').split(',')
        authorsExt = [a.strip() for a in authorsExt]
        pdf = paperInfo.find('li').a.get('href')

        info = {'title': title.get_text(),
                'abstract': abstract.get_text(),
                'authors': authorsExt,
                'pdf': pdf,
                'keywords': []}
        print(cnt)
        fullDB[title.get_text()] = [info, 'Accept', venue]
        cnt+=1
        #if cnt >= 50:
        #    break

#initDB_icml(icml2018url, fullDB, "icml2018")

#for k, v in fullDB.items():
#    print(v)
#    break
    
#print(soup.find_all('li')[1].a.get('href'))
