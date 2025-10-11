import os
import re
import numpy as np
import json

def doc_grouper(folder_dir):
    base_dir=folder_dir
    docs=[]
    tag="doc"
    for file in os.listdir(folder_dir):
        file_path=os.path.join(base_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content=f.read()
            reg_str = r"<doc[^>]*>(.*?)</doc>"
            res = re.findall(reg_str, content, flags=re.S)
            docs.append(res)
    with open("results.json", 'w') as file:
        json.dump(docs, file, ensure_ascii=False)
    return docs
 

def sectionizer(folder_dir):
    docs = doc_grouper(folder_dir)
    articles=[]
    for txt_file in docs:
        for article in txt_file: #each array is an entire article
            reg_str = r"\n(.*?)\n"
            res = re.findall(reg_str, article, flags=re.S)
            if len(res) < 3:
                continue
            else:
                articles.append(res)
    return articles

def dictionary(folder_dir):
    categorized=[]
    articles = sectionizer(folder_dir)
    for a in articles:
        art = {}
        key=a[0]
        title=key
        art[f"{key}"] = a[0]
        ind=1
        while ind<len(a):
            if len(a[ind].split()) < 3 and a[ind] != "":
                key = a[ind][0:len(a[ind])-1]
                check_ind=ind+1
                print("hit")
                if check_ind < len(a) and a[check_ind] != "":
                    art[f"{title}, {key}"] = ""
                    print("hit1")
                elif check_ind < len(a) and a[check_ind] == "":
                    ind+=1
                    print("hit2")
                    continue
                else:
                    ind+=1
                    continue
            elif a[ind] != "":
                print("hit3")
                if key == title:
                    art[f"{title}"]+= " " + a[ind]
                else:
                    art[f"{title}, {key}"]+=" " + a[ind]
            ind+=1
        empty_keys=[k for k, v in art.items() if not v]
        for k in empty_keys:
            del art[k]
        categorized.append(art)
    with open("categorized.json", 'w', encoding='utf-8') as f:
        json.dump(categorized, f, ensure_ascii=False, indent=4)
    return categorized
        

if __name__ == "__main__":
    dictionary("./text")