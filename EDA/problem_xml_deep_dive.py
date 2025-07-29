from lxml import etree

# import data
root = etree.parse('data/congress_119_hr1/problem xml.xml')

for element in root.iter():
    # Get the XPath for the current element
    xpath = root.getpath(element)
    #print(f"Element: <{element.tag}>, XPath: {xpath}")
    print(xpath)
# for all of these, paragraph shows up multiple times

# idea: for paragraph id H55756362A59148379834D71CAD3B8F76, break it up further into paragraphs
# or maybe check if paragraph in paragraphs?    

# do path eda, count how many times paragraph shows up in the path
