import os

# === Variables to change ===
language = 'German'
# ===========================


def create_empty_files(language='English'):
    query_tag_list = []
    query_tag_list += [f'b{num+1}' for num in range(7)]
    query_tag_list += [f'd{num+1}' for num in range(2)]
    query_tag_list += [f'h{num+1}' for num in range(3)]
    query_tag_list += [f'o{num+1}' for num in range(8)]

    print(query_tag_list)
    
    os.makedirs(os.path.join('0-data', 'lang_documents', language))

    for query_tag in query_tag_list:
        fpath = os.path.join('0-data', 'lang_documents', language, f'{query_tag}.txt')
        print(fpath)
        if not os.path.exists(fpath):
            open(fpath, 'w').close()
            
create_empty_files(language)
