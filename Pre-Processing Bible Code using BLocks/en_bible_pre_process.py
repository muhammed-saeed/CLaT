#!/bin/env python3

import fitz  # install using: pip install PyMuPDF
# reads the pdf files
import re
import os
import sys
import json
from collections import defaultdict

number_pattern = re.compile("\d+")


def main():
    bible = fitz.open("en_bible.pdf")
    toc = bible.get_toc()
    num_chapters = len(toc)

    books = []
    for i, (_, _, start) in enumerate(toc):
        end = toc[i+1][2] - 1 if i < num_chapters - 1 else None
        chapters = process_book(bible, start - 1, end)
        books.append((i+1, chapters))
    write_bible(books)


def process_book(bible, start, end):
    chapters = defaultdict(list)
    chapter = 1
    verse = ""
    has_skipped_title = False
    for page in bible.pages(start, end):
        # skip the heading in all pages
        blocks = page.get_text("dict")["blocks"][1:]
        # the pdf is consist of blocks and the blocks into span and into text
        for block in blocks:
            for line in block["lines"]:
                for span in line["spans"]:
                    # span contains (text and color)
                    text = span["text"].strip()
                    color = span["color"]
                    if color == 128:
                        # 128 is blue color verse number
                        has_skipped_title = True

                    if has_skipped_title:
                        # because each chapter has introduction and title and then co
                        # the title and the introduciton color is blue and the
                        if number_pattern.fullmatch(text):
                            # for the second verse we have, verse for the first one,
                            if verse:
                                # for the first verse in each chapter will start
                                chapters[chapter].append(verse)
                            verse = ""

                            if color == 0:  # isolated black numbers are chapter numbers
                                chapter += 1
                                # if the span color is black then add the verse to the chapter and change the chapter to the new number
                        else:
                            if verse:
                                verse += " "
                            verse += text

    if verse:
        # the final verse is not followed by verse_number and the verse is added with the followiing_verse_number
        chapters[chapter].append(verse)
    return chapters


def write_bible(books):
    for i, chapters in books:
        for j, chapter in chapters.items():
            with open(f"en/{i}_{j}.txt", "w") as f:
                f.write("\n".join(chapter))


if __name__ == "__main__":
    main()
