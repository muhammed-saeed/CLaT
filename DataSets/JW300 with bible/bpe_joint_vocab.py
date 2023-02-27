import sentencepiece as spm
en_file_path = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\train\train.en"
pcm_file_path = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\train\train.pcm"
dict_path = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\bpe_dict_path"

joint_vocab_file = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\joint_train.txt"

en_train_input = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\train\train.en"
en_train_bpe_output = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\train_bpe\train.bpe.en"
pcm_train_input = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\train\train.pcm"
pcm_train_bpe_output = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\train_bpe\train.bpe.pcm"

en_val_input = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\val\val.en"
en_val_bpe_output = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\val_bpe\val.bpe.en"
pcm_val_input = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\val\val.pcm"
pcm_val_bpe_output = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\val_bpe\val.bpe.pcm"

en_test_input = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\test\test.en"
en_test_bpe_output = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\test_bpe\test.bpe.en"
pcm_test_input = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\test\test.pcm"
pcm_test_bpe_output = r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\test_bpe\test.bpe.pcm"

vocab_size = 15000

pcm_source_file = ""
en_source_file = ""

with open(en_file_path, "r", encoding="utf-8") as fb:
    en_source_file = fb.read()

with open(pcm_file_path, "r", encoding="utf-8") as fb:
    pcm_source_file = fb.read()

with open(joint_vocab_file, "w", encoding="utf-8") as fb:
    fb.write(en_source_file)
    fb.write("\n")
    fb.write(pcm_source_file)


def train_en(vocab_size):
  model_prefix = dict_path+"\en_" + "_vocab_" + str(vocab_size)
  spm.SentencePieceTrainer.train(input=joint_vocab_file
      , model_prefix=model_prefix
      , vocab_size=vocab_size
      , character_coverage = 0.9995
      , num_threads=60
      , model_type = "bpe"
      , train_extremely_large_corpus=True
  )
train_en(vocab_size)

def train_pcm(vocab_size):
  model_prefix = dict_path + "\pcm_" + "_vocab_" + str(vocab_size)
  spm.SentencePieceTrainer.train(input=joint_vocab_file
      , model_prefix=model_prefix
      , vocab_size=vocab_size
      , character_coverage = 0.9995
      , num_threads=60
      ,model_type = "bpe"
      , train_extremely_large_corpus=True
  )
train_pcm(vocab_size)

en_tokenizer = spm.SentencePieceProcessor(model_file=r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\bpe_dict_path\en__vocab_15000.model")
pcm_tokenizer = spm.SentencePieceProcessor(model_file=r"C:\Users\lst\Desktop\Naija-Pidgin\JW300 with bible\bpe_dict_path\pcm__vocab_15000.model")


with open(en_train_input, "r", encoding="utf-8") as rf, open(en_train_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(en_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))

with open(pcm_train_input, "r", encoding="utf-8") as rf, open(pcm_train_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(pcm_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))



with open(en_test_input, "r", encoding="utf-8") as rf, open(en_test_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(en_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))


with open(pcm_test_input, "r", encoding="utf-8") as rf, open(pcm_test_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(pcm_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))



with open(en_val_input, "r", encoding="utf-8") as rf, open(en_val_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(en_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))



with open(pcm_val_input, "r", encoding="utf-8") as rf, open(pcm_val_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(pcm_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))

