from lxml import etree
import pandas as pd

# import data
root = etree.parse('data/congress_119_hr1/BILLS-119hr1eh.xml')

paths_counts = []
for element in root.iter():
    # Get the XPath for the current element
    if len(element)==0:
        res_dict = dict()
        path = root.getpath(element)
        res_dict['path'] = path
        res_dict['paragraph_count'] = path.count('/paragraph')
        res_dict['subsection_count'] = path.count('/subsection')
        res_dict['section_count'] = path.count('/section')
        res_dict['all_count'] = res_dict['paragraph_count'] + res_dict['subsection_count'] + res_dict['section_count']
        paths_counts.append(res_dict)

paths_counts_df = pd.DataFrame(paths_counts)

with open("EDA/paths.txt", "w") as f_out:
    f_out.write("\n".join(paths_counts_df["path"]))

print(paths_counts_df.head())
print(paths_counts_df.info())

print(paths_counts_df[['paragraph_count', 'subsection_count', 'section_count']].value_counts())


print(paths_counts_df[paths_counts_df['all_count'] > 3]['path'][557])