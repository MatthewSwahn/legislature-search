from lxml import etree
from typing import Optional, List, Dict

LOWEST_TAGS = {'section', 'subsection', 'paragraph', 'subparagraph'}

def is_lowest_level(elem):
    tag = elem.tag.split('}', 1)[-1]
    return tag in LOWEST_TAGS

def get_xpath(elem):
    path = []
    while elem is not None and elem.getparent() is not None:
        parent = elem.getparent()
        tag = elem.tag.split('}', 1)[-1]
        siblings = [sib for sib in parent if sib.tag == elem.tag]
        if len(siblings) > 1:
            ix = siblings.index(elem) + 1
            path.append(f'{tag}[{ix}]')
        else:
            path.append(tag)
        elem = parent
    path.reverse()
    return '/' + '/'.join(path)

def get_direct_text(elem):
    text_chunks = []
    if elem.text and elem.text.strip():
        text_chunks.append(elem.text.strip())
    for child in elem:
        tag = child.tag.split('}', 1)[-1]
        if tag not in LOWEST_TAGS:
            if child.text and child.text.strip():
                text_chunks.append(child.text.strip())
            for subchild in child:
                if subchild.text and subchild.text.strip():
                    text_chunks.append(subchild.text.strip())
    return ' '.join(text_chunks)

def should_include(xpath):
    # Exclude unwanted paths
    return not (
        xpath.startswith('/bill/metadata/') or
        xpath.startswith('/bill/form/') or
        '/toc/' in xpath
    )

def parse_xml(root: etree.Element) -> List[Dict]:
    rows = []

    for elem in root.iter():
        if is_lowest_level(elem):
            xpath = get_xpath(elem)
            if should_include(xpath):
                row = {
                    'xpath': xpath,
                    'tag': elem.tag.split('}', 1)[-1],
                    'text': get_direct_text(elem),
                }
                rows.append(row)

    return rows
