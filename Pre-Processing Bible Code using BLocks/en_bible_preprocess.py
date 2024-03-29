#!/bin/env python3

import fitz
import re
import os
import sys
import json
import argparse
from collections import defaultdict

number_pattern = re.compile("\d+")


def main(pdf_path, output_dir):
    bible = fitz.open(pdf_path)
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

                            if color == 0:
                                chapter += 1
                        else:
                            if verse:
                                verse += " "
                            verse += text

    if verse:
        chapters[chapter].append(verse)
    return chapters


def write_bible(books, output_dir):
    for i, chapters in books:
        for j, chapter in chapters.items():
            with open(f"{output_dir}/en_{i}_{j}.txt", "w") as f:
                f.write("\n".join(chapter))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a bible pdf file.")
    parser.add_argument("pdf_path", help="Path to the pdf file to process.")
    parser.add_argument("output_dir", help="Path to the output directory.")
    args = parser.parse_args()

    main(args.pdf_path, args.output_dir)
