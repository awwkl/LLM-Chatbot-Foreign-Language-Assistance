import os
from bardapi import Bard
from bardapi import BardCookies

# === Variables to change ===
# Spanish, Mandarin, French, German, and Italian
language_list = ['French', 'German', 'Italian', 'Mandarin', 'Spanish']

query_tags_to_ignore = ['b1']
# ===========================

# === Basic setup ===
# Any cookie values you want to pass session object.
cookie_dict = {
    # "__Secure-1PSID": "___",
    # "__Secure-1PSIDTS": "___"
}
bard = BardCookies(cookie_dict=cookie_dict)

for language in language_list:
    os.makedirs(os.path.join('0-data', 'lang_documents', language), exist_ok=True)
# ===========================

# token = ""
# bard = Bard(token=token)

# query_to_bard = "How to thank someone? (in English, for someone new to the English language). Give me examples and format each example like this: English words: (English translation, English pronunciation). Use point format, and not table format. Afterwards, give me some tips specific to the question I asked, not generic tips for learning English. "
# bard_answer = bard.get_answer(query_to_bard)['content']
# print(bard_answer)

def create_queries_for_bard(lang='English'):
    
    append_line = f'(in {lang}, for someone new to the {lang} language). Give me examples and format each example like this: {lang} words: (English translation, English pronunciation). Do not forget the pronunciation. Use point format, and not table format. Afterwards, give me some tips specific to the question I asked, not generic tips for learning {lang}. '
    
    query_file_path = '0-data/train_queries.txt'
    query_file = open(query_file_path, 'r')
    for line in query_file.readlines():
        if not line.startswith('['):
            continue
        
        query_tag = line[:4].replace('[', '').replace(']', '')
        fpath = os.path.join('0-data', 'lang_documents', language, f'{query_tag}.txt')
        if os.path.exists(fpath): # if file exists, continue
            continue

        query = line.split('] ')[1].strip()
        query = query.split('||')[0].strip()
        query_to_bard = f'{query} {append_line}'
        
        print(query_tag)
        print(query_to_bard)
        
        bard_answer = bard.get_answer(query_to_bard)['content']
        bard_answer = bard_answer.replace('**', '') # preprocessing
        
        if 'Response Error' in bard_answer:
            print('[DEBUG] Error.')
            continue
        
        # === Write answer to file ===
        f = open(fpath, 'w')
        f.write(bard_answer)
        f.close()

for language in language_list:
    create_queries_for_bard(language)
