# driver.py
# Luke Reichold - CSCI 4930
from summarize import Summarizer


argentina_articles = ["argentina/argentina-guardian.txt", "argentina/argentina-nyt.txt"]
china_articles = ["china/china-cnn.txt", "china/china-nyt.txt"]
climate_articles = ["climate/climate-npr.txt", "climate/climate-nyt.txt"]
VW_articles = ["VW/VW-ars.txt", "VW/VW-nyt.txt"]


magic = Summarizer(VW_articles)
print(magic.generate_summaries())
