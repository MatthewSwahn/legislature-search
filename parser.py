from lxml import etree
from typing import Optional, List, Dict

def _get_id(elem) -> Optional[str]:
    if elem is None:
        return None
    return elem.get('id')

def _find_parent_with_tag(elem: etree.Element, tagname: str) -> Optional[etree.Element]:
    parent = elem.getparent()
    while parent is not None:
        if parent.tag == tagname:
            return parent
        parent = parent.getparent()
    return None

def _find_nearest_quoted_block(elem: etree.Element) -> Optional[etree.Element]:
    return _find_parent_with_tag(elem, 'quoted-block')

def _find_quoted_block_ancestor(elem: etree.Element, tagname: str) -> Optional[etree.Element]:
    """
    Find the nearest ancestor with given tag **inside the same quoted-block**.
    """
    qb = _find_nearest_quoted_block(elem)
    parent = elem.getparent()
    while parent is not None and parent is not qb:
        if parent.tag == tagname:
            return parent
        parent = parent.getparent()
    return None

def _get_quoted_block_path(root: etree.Element, elem: Optional[etree.Element]) -> Optional[str]:
    if elem is None:
        return None
    return root.getpath(elem)

def parse_xml(root: etree.Element) -> List[Dict]:
    """
    Groups text at the deepest available level, including clause (and subclause) in both normal and quoted-block context.
    Each result includes normal and quoted-block ancestry fields.
    """
    groups = []

    # 1. Group by clause (deepest, including within quoted-blocks)
    for clause in root.xpath('.//clause'):
        if not isinstance(clause, etree._Element):
            continue
        # Normal tree
        subparagraph = _find_parent_with_tag(clause, 'subparagraph')
        paragraph = _find_parent_with_tag(clause, 'paragraph')
        subsection = _find_parent_with_tag(clause, 'subsection')
        section = _find_parent_with_tag(clause, 'section')
        # Quoted-block tree
        quoted_block = _find_nearest_quoted_block(clause)
        qb_subparagraph = _find_quoted_block_ancestor(clause, 'subparagraph')
        qb_paragraph = _find_quoted_block_ancestor(clause, 'paragraph')
        qb_subsection = _find_quoted_block_ancestor(clause, 'subsection')
        qb_section = _find_quoted_block_ancestor(clause, 'section')
        text = ''.join(clause.itertext()).strip()
        if text:
            groups.append({
                'level': 'clause',
                'clause_id': _get_id(clause),
                'subclause_id': None,
                'subparagraph_id': _get_id(subparagraph),
                'paragraph_id': _get_id(paragraph),
                'subsection_id': _get_id(subsection),
                'section_id': _get_id(section),
                'quoted_block_subclause_id': None,
                'quoted_block_subparagraph_id': _get_id(qb_subparagraph),
                'quoted_block_paragraph_id': _get_id(qb_paragraph),
                'quoted_block_subsection_id': _get_id(qb_subsection),
                'quoted_block_section_id': _get_id(qb_section),
                'quoted_block_id': _get_id(quoted_block),
                'quoted_block_path': _get_quoted_block_path(root, quoted_block),
                'path': root.getpath(clause),
                'text': text,
            })

    # 2. Subparagraphs (only those that do not contain clauses)
    for subparagraph in root.xpath('.//subparagraph'):
        if not isinstance(subparagraph, etree._Element) or subparagraph.xpath('./clause'):
            continue
        # ... (same as before, see previous code)
        # [rest of your subparagraph logic here]

    # 3. Paragraphs (only those that do not contain subparagraphs or clauses)
    for paragraph in root.xpath('.//paragraph'):
        if (not isinstance(paragraph, etree._Element) or
            paragraph.xpath('./subparagraph') or
            paragraph.xpath('./clause')):
            continue
        # ... (same as before, see previous code)
        # [rest of your paragraph logic here]

    # 4. Subsections (only those that do not contain paragraphs, subparagraphs, or clauses)
    for subsection in root.xpath('.//subsection'):
        if (not isinstance(subsection, etree._Element) or
            subsection.xpath('./paragraph') or
            subsection.xpath('./subparagraph') or
            subsection.xpath('./clause')):
            continue
        # ... (same as before, see previous code)
        # [rest of your subsection logic here]

    # 5. Sections (only those that do not contain subsections, paragraphs, subparagraphs, or clauses)
    for section in root.xpath('.//section'):
        if (not isinstance(section, etree._Element) or
            section.xpath('./subsection') or
            section.xpath('./paragraph') or
            section.xpath('./subparagraph') or
            section.xpath('./clause')):
            continue
        # ... (same as before, see previous code)
        # [rest of your section logic here]

    return groups