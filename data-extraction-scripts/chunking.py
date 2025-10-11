import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk():
    chunk_id=0
    result=[]
    chunk_size=500
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size/10),
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    with open("categorized.json", 'r') as f:
        data=json.load(f)
    for article in data:
        keys=iter(article)
        key=next(keys, "end")
        title=key
        while True:
            if key=="end":
                break
            s=article[key]
            splits=text_splitter.split_text(s)
            for x in splits:
                chunk_dict={}
                chunk_dict["ID"]=chunk_id
                chunk_dict["Title"]=title
                chunk_dict["Section"]=key
                chunk_dict["Content"]=x
                chunk_id+=1
                result.append(chunk_dict)
            key=next(keys, "end")

    with open("chunks.json", 'w') as f:
        print("hit")
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    chunk()