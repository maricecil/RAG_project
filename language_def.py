from openai import OpenAI
client = OpenAI()
completion2 = client.chat.completions.create(
    model = 'gpt-4o',
    messages = [
        {'role' : 'system', 'content' : 'translate the given content into {language}'},
        {'role' : 'user', 'content': }
    ],
    temperature = 0,
    max_tokens= 1000
    )

def language(language_selection):
    if language_selection == 1 or '일본어' or 'Japanses':
        translated_sentence = completion2.format(language = 'Japanses')
    elif language_selection == 1 or '영어' or 'English':
        translated_sentence = completion2.format(language = 'English')
    elif language_selection == 1 or '한국어' or 'Korean':
        translated_sentence = completion2.format(language = 'Korean')
    elif language_selection == 1 or '프랑스어' or 'French':
        translated_sentence = completion2.format(language = 'French')
    elif language_selection == 1 or '독일어' or 'Germany':
        translated_sentence = completion2.format(language = 'Germany')
    elif language_selection == 1 or '러시아어' or 'Russian':
        translated_sentence = completion2.format(language = 'Russian')
    elif language_selection == 1 or '스페인어' or 'Spanish':
        translated_sentence = completion2.format(language = 'Spanish')
    else:
        translated_sentence = completion2.format(language = language_selection)
        return translated_sentence
    
while True:
    language_selection = input('''언어를 선택하세요:
                           1. 일본어(Japanses)
                           2. 영어(English)
                           3. 한국어(Korean)
                           4. 프랑스어(French)
                           5. 독일어(Germany)
                           6. 러시아어(Russian)
                           7. 스페인어(Spanish)
                           혹은 원하는 언어를 적어주세요: ''')
    try:
        print(language(language_selection))
    except Exception as e:
        print('죄송합니다. 작성하신 언어는 존재하지 않거나, 지원하지 않습니다. 언어를 다시 확인해 주세요')