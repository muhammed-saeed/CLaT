from cgitb import text
import csv

csv_file = "C:/Users/lst/Desktop/Naija-Pidgin/bbc_scrapper/bbc_scrapper/bbc_pidgin_corpus.csv"
text_file = "C:/Users/lst/Desktop/Naija-Pidgin/bbc_scrapper/bbc_scrapper/text_data_.txt"

with open(text_file, "w") as my_output_file:
    with open(csv_file, "r", encoding="utf8") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()
lines = ""
with open(text_file, "r") as fb:
    lines = fb.readlines()
print(len(lines))

text_file_no_blank_lines = r"C:\Users\lst\Desktop\Naija-Pidgin\Others Dataset\other mono-pcm\bbc_scrapper\bbc_scrapped_data.txt"

with open(text_file_no_blank_lines, "w", encoding="utf-8") as fb:
    for line in lines:
        line = line.strip()
        line = line.strip()
        fb.write(line)
        
        