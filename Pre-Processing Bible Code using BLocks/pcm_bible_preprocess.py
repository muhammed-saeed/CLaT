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
    bible = fitz.open("pcm_bible.pdf")
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
        for block in blocks:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    color = span["color"]

                    if color == 128:
                        has_skipped_title = True

                    if has_skipped_title:
                        if number_pattern.fullmatch(text):
                            if verse:
                                chapters[chapter].append(verse)
                            verse = ""

                            if color == 0:  # isolated black numbers are chapter numbers
                                chapter += 1
                        elif has_skipped_title and color == 0:  # only accept black text as verses
                            # the introduciton is not number but colored is blue
                            # and also the blue text above each chapter
                            if verse:
                                verse += " "
                            verse += text

    if verse:
        chapters[chapter].append(verse)
    return chapters


def write_bible(books):
    for i, chapters in books:
        for j, chapter in chapters.items():
            with open(f"pcm/{i}_{j}.txt", "w") as f:
                f.write("\n".join(chapter))


if __name__ == "__main__":
    main()
