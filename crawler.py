#!/usr/bin/env python

from bs4 import BeautifulSoup
import requests

url = "https://www.totalwine.com/wine-brands"
response = requests.get(url)
page = BeautifulSoup(response.text, "html.parser")
elements = page.select("div > ul > li > span > a")
file = open("names.txt", "w")
for e in elements:
	text = e.get_text()
	print(text)
	file.write("%s\n" % text.encode("utf-8"))
file.close()
