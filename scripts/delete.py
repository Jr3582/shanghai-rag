import fitz
import os
import re

doc1 = fitz.open("../data/battle_of_shanghai_wikipedia.pdf")
doc1.delete_pages(from_page=30, to_page=41)
doc1.save("../data/battle_of_shanghai_wikipedia_TRIMMED.pdf")
doc2 = fitz.open("../data/Defense of Sihang Warehouse - Wikipedia.pdf")
doc2.delete_pages(from_page=22, to_page=30)
doc2.save("../data/Defense of Sihang Warehouse - Wikipedia_TRIMMED.pdf")
doc3 = fitz.open("../data/Second Sino-Japanese War - Wikipedia.pdf")
doc3.delete_pages(from_page=35, to_page=54)
doc3.save("../data/Second Sino-Japanese War - Wikipedia_TRIMMED.pdf")
doc4 = fitz.open("../data/Battle of Shanghai 1937 - Pacific Atrocities Education.pdf")
doc4.delete_pages(from_page=4, to_page=7)
doc4.save("../data/Battle of Shanghai 1937 - Pacific Atrocities Education_TRIMMED.pdf")