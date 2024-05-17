import numpy as np
import pandas as pd
from lxml import etree
from web_utils import beautify_event_table, beautify_rank_table, determine_next_matchday

next_matchday = determine_next_matchday()

# Read and clean data tables
df_event = pd.read_csv(
    f"./data/event_CSVs/event_matchday_{next_matchday}.csv", index_col=0
)
df_rank = pd.read_csv(f"./data/ranks/ranks_matchday_{next_matchday}.csv", index_col=0)

df_event = beautify_event_table(df_event)
df_rank = beautify_rank_table(df_rank, next_matchday)

# Generate HTML for matches table
matches_html = df_event.to_html(index=False)

# Generate HTML for team rankings table
team_rankings_html = df_rank.to_html(index=False)

# Create HTML structure using lxml etree
html = etree.Element("html")

# Add DOCTYPE and html lang attribute
doctype = "<!DOCTYPE html>\n"
html.set("lang", "en")

# Create head element
head = etree.SubElement(html, "head")
title = etree.SubElement(head, "title")
title.text = "Sigma Stats"
meta_charset = etree.SubElement(head, "meta", charset="UTF-8")
meta_viewport = etree.SubElement(
    head, "meta", name="viewport", content="width=device-width, initial-scale=1.0"
)
meta_ie_compat = etree.SubElement(
    head, "meta", http_equiv="X-UA-Compatible", content="ie=edge"
)
link = etree.SubElement(
    head, "link", rel="stylesheet", type="text/css", href="styles.css"
)

# Create body element
body = etree.SubElement(html, "body")

# Create header
header = etree.SubElement(body, "header")
header_h1 = etree.SubElement(header, "h1")
header_h1.text = "Canadian Premier League Power Rankings & Match Predictions"

# Create matches section
matches_section = etree.SubElement(body, "section", id="matches")
matches_header = etree.SubElement(matches_section, "h2")
matches_header.text = "ELO v1 Match Predictions"
matches_table = etree.SubElement(matches_section, "table")
matches_table.append(etree.fromstring(matches_html))

# Create team rankings section
team_rankings_section = etree.SubElement(body, "section", id="team-rankings")
team_rankings_header = etree.SubElement(team_rankings_section, "h2")
team_rankings_header.text = "ELO v1 Team Rankings"
team_rankings_table = etree.SubElement(team_rankings_section, "table")
team_rankings_table.append(etree.fromstring(team_rankings_html))

# Serialize HTML tree to a string with indentation
html_str = etree.tostring(html, pretty_print=True, encoding="unicode")

# Write HTML content to a file
with open("index.html", "w") as file:
    file.write(doctype)
    file.write(html_str)
