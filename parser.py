from lxml import etree
from typing import Optional


def get_id(elem) -> Optional[etree.Element]:
    """Return the id attribute if present, else None."""
    if elem is None:
        return None
    return elem.get('id')

def find_parent_with_tag(elem:etree.Element, tagname:str) -> Optional[etree.Element]:
    """Walk up the tree from elem and return first parent with the given tag."""
    parent = elem.getparent()
    while parent is not None:
        if parent.tag == tagname:
            return parent
        parent = parent.getparent()
    return None


def group_legal_text_paragraph_level(root:etree.Element) -> list:
    """
    Groups text at the deepest available level:
    paragraph > subsection > section.
    Ignores subparagraphs as grouping levels.
    Each result includes paragraph_id, subsection_id, section_id, XPath, and text.
    """
    groups = []
    # Group by paragraph
    for paragraph in root.xpath('.//paragraph'):
        subsection = find_parent_with_tag(paragraph, 'subsection')
        section = find_parent_with_tag(paragraph, 'section')
        text = ''.join(paragraph.itertext()).strip()
        if text:
            groups.append({
                'level': 'paragraph',
                'subparagraph_id': None,  # Not grouping by subparagraphs
                'paragraph_id': get_id(paragraph),
                'subsection_id': get_id(subsection),
                'section_id': get_id(section),
                'path': root.getpath(paragraph),
                'text': text,
            })

    # Group by subsection if no paragraphs
    for subsection in root.xpath('.//subsection'):
        # Only include subsections without paragraphs
        if not subsection.xpath('./paragraph'):
            section = find_parent_with_tag(subsection, 'section')
            text = ''.join(subsection.itertext()).strip()
            if text:
                groups.append({
                    'level': 'subsection',
                    'subparagraph_id': None,
                    'paragraph_id': None,
                    'subsection_id': get_id(subsection),
                    'section_id': get_id(section),
                    'path': root.getpath(subsection),
                    'text': text,
                })

    # Group by section if no subsections or paragraphs
    for section in root.xpath('.//section'):
        # Only include sections without subsections or paragraphs
        if not section.xpath('./subsection') and not section.xpath('./paragraph'):
            text = ''.join(section.itertext()).strip()
            if text:
                groups.append({
                    'level': 'section',
                    'subparagraph_id': None,
                    'paragraph_id': None,
                    'subsection_id': None,
                    'section_id': get_id(section),
                    'path': root.getpath(section),
                    'text': text,
                })

    return groups
