pcm_bible = r"C:\Users\lst\Desktop\Naija-Pidgin\BLOCKS_SPANS\BLOCKS_SPANS\pcm_bible\pcm_entire_bible_2.txt"
en_bible = r"C:\Users\lst\Desktop\Naija-Pidgin\BLOCKS_SPANS\BLOCKS_SPANS\en_bible\en_entire_bible_2.txt"

train_jw300_en = r"C:\Users\lst\Desktop\masakhane_benchmarks\en-pcm\jw300-baseline\data\train.en"
train_jw300_pcm = r"C:\Users\lst\Desktop\masakhane_benchmarks\en-pcm\jw300-baseline\data\train.pcm"
test_jw300_en = r"C:\Users\lst\Desktop\masakhane_benchmarks\en-pcm\jw300-baseline\data\test.en"
test_jw300_pcm = r"C:\Users\lst\Desktop\masakhane_benchmarks\en-pcm\jw300-baseline\data\test.pcm"
dev_jw300_en = r"C:\Users\lst\Desktop\masakhane_benchmarks\en-pcm\jw300-baseline\data\dev.en"
dev_jw300_pcm = r"C:\Users\lst\Desktop\masakhane_benchmarks\en-pcm\jw300-baseline\data\dev.pcm"

our_en = []
our_pcm = []
jw300_en = []
jw_300_pcm = []
with open(train_jw300_en, "r", encoding="utf-8") as fb:
    jw300_en.extend(fb.readlines())

with open(train_jw300_pcm, "r", encoding="utf-8") as fb:
    jw_300_pcm.extend(fb.readlines())

test_en_data = []
test_pcm_data = []
# with open(test_jw300_en, "r", encoding="utf-8") as fb:
#     jw300_en.extend(fb.readlines())
with open(test_jw300_en, "r", encoding="utf-8") as fb:
    test_en_data = fb.readlines()
# with open(test_jw300_pcm, "r", encoding="utf-8") as fb:
#     jw_300_pcm.extend(fb.readlines())
with open(test_jw300_pcm, "r", encoding="utf-8") as fb:
    test_pcm_data = fb.readlines()


with open(dev_jw300_en, "r", encoding="utf-8") as fb:
    jw300_en.extend(fb.readlines())

with open(dev_jw300_pcm, "r", encoding="utf-8") as fb:
    jw_300_pcm.extend(fb.readlines())

print(f"length of en {len(jw300_en)} and pcm is {len(jw_300_pcm)}")

with open(pcm_bible, "r", encoding="utf-8") as fb:
    our_pcm.extend(fb.readlines())

with open(en_bible, "r", encoding="utf-8") as fb:
    our_en.extend(fb.readlines())

our_pcm.extend(jw_300_pcm)
our_en.extend(jw300_en)

print(f"the english lenght is {len(our_en)} and pcm is {len(our_pcm)}")

new_en = r"C:\Users\lst\Desktop\Naija-Pidgin\BLOCKS_SPANS\BLOCKS_SPANS\en_bible\en_entire_bible_jw300.txt"
new_pcm= r"C:\Users\lst\Desktop\Naija-Pidgin\BLOCKS_SPANS\BLOCKS_SPANS\pcm_bible\pcm_entire_bible_jw300.txt"

with open(new_en, 'w', encoding="utf-8") as fb:
    fb.writelines(our_en)

with open(new_pcm, 'w', encoding="utf-8") as fb:
    fb.writelines(our_pcm)

import random
 
# initializing lists
# test_list1 = [6, 4, 8, 9, 10]
# test_list2 = [1, 2, 3, 4, 5]
 
# # printing lists
# print(f"The original list 1 : {test_list1}")
# print(f"The original list 2 : {test_list2}")
 
# Shuffle two lists with same order
# Using zip() + * operator + shuffle()
temp = list(zip(our_pcm, our_en))
random.shuffle(temp)
res1, res2 = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
our_pcm, our_en = list(res1), list(res2)

# entire_Data = r"C:\Users\lst\Desktop\Naija-Pidgin\Bible Processing Using Blocks Method\Machine Translation Data (FairSeq)\text_files\pcm_shuffled.txt"
# en_shuffled = r"C:\Users\lst\Desktop\Naija-Pidgin\Bible Processing Using Blocks Method\Machine Translation Data (FairSeq)\text_files\en_shuffled.txt"

# with open(pcm_shuffled, "w", encoding="utf-8") as fb:
#     fb.writelines(our_pcm)

# with open(en_shuffled, "w", encoding="utf-8") as fb:
#     fb.writelines(our_en)




train_en = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\train\train.en"
train_pcm = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\train\train.pcm"

test_en = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\test\test.en"
test_pcm = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\test\test.pcm"

val_en = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\val\val.en"
val_pcm = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\val\val.pcm"

with open(train_en, "w", encoding="utf-8") as fb:
    fb.writelines(our_en[:-3200])


with open(train_pcm, "w", encoding="utf-8") as fb:
    fb.writelines(our_pcm[:-3200])


with open(val_en, "w", encoding="utf-8") as fb:
    fb.writelines(our_en[-2200:])



with open(val_pcm, "w", encoding="utf-8") as fb:
    fb.writelines(our_pcm[-2200:])


with open(test_en, "w", encoding="utf-8") as fb:
    # fb.writelines(our_en[-2200:])
    fb.writelines(test_en_data)


with open(test_pcm, "w", encoding="utf-8") as fb:
    # fb.writelines(our_pcm[-2200:])
    fb.writelines(test_pcm_data)