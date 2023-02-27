train= r"C:\Users\lst\Desktop\Naija-Pidgin\bbc_scrapper\AFRI_BERT Mono-Data\train.txt"
data_1 = ""
eval = r"C:\Users\lst\Desktop\Naija-Pidgin\bbc_scrapper\AFRI_BERT Mono-Data\eval.txt"
data_2 = ""
with open(train, "r", encoding="utf8") as fb:
    data_1 = fb.read()
with open(eval, "r", encoding="utf8") as fb:
    data_2 = fb.read()

total = data_1 + "\n" + data_2
merged_data =  r"C:\Users\lst\Desktop\Naija-Pidgin\bbc_scrapper\AFRI_BERT Mono-Data\AFRI_BERT_MONO.txt"

with open(merged_data, "w", encoding="utf-8") as fb:
    fb.write(total)
