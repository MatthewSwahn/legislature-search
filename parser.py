from lxml import etree

def get_section_id(section):
    section_id = section.get('id')
    if section_id:
        return section_id
    id_elem = section.find('./id')
    if id_elem is not None and id_elem.text:
        return id_elem.text.strip()
    return None

def get_xml_text_by_paragraph(root:etree.Element) -> list:
    groups = []
    for section in root.xpath('.//section'):
        section_id = get_section_id(section)
        paragraphs = section.xpath('./paragraph')
        if paragraphs:
            for paragraph in paragraphs:
                text = ' '.join(paragraph.itertext()).strip()
                if text:
                    groups.append({
                        'level': 'paragraph',
                        'section_id': section_id,
                        'path': root.getpath(paragraph),
                        'text': text,
                    })
        else:
            text = ' '.join(section.itertext()).strip()
            if text:
                groups.append({
                    'level': 'section',
                    'section_id': section_id,
                    'path': root.getpath(section),
                    'text': text,
                })
    return groups

def group_section_paragraphs_with_id_and_path(root:etree.Element) -> list:
    groups = []
    tree = root.getroottree()
    for section in root.xpath('.//section'):
        section_id = get_section_id(section)
        paragraphs = section.xpath('./paragraph')
        if paragraphs:
            for paragraph in paragraphs:
                text = ''.join(paragraph.itertext()).strip()
                if text:
                    groups.append({
                        'level': 'paragraph',
                        'section_id': section_id,
                        'path': tree.getpath(paragraph),
                        'text': text,
                    })
        else:
            text = ''.join(section.itertext()).strip()
            if text:
                groups.append({
                    'level': 'section',
                    'section_id': section_id,
                    'path': tree.getpath(section),
                    'text': text,
                })
    return groups