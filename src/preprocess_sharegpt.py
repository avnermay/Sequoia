# NOTE: First need to download dataset:
# wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json .

import jsonlines

import random
from transformers import AutoTokenizer
from datasets import load_dataset


def convert_turn_format(turn):
    return {
        'role': 'user' if turn['from'] == 'human' else 'assistant',
        'content': turn['value'],
    }


# Extract all turns.
def write_all_turns(chats, output_file):
    with open(output_file, 'w') as f_out:
        writer = jsonlines.Writer(f_out)
        for chat in chats:
            chat_in_llama3_format = tokenizer.apply_chat_template(chat, tokenize=False)
            new_item = {'text': chat_in_llama3_format}
            writer.write(new_item)
        writer.close()

    print(f'Finished writing {output_file}')


# Extract just first prompt and response from ChatGPT.
def write_prompt_response(chats, output_file):
    with open(output_file, 'w') as f_out:
        writer = jsonlines.Writer(f_out)
        for chat in chats:
            if len(chat) >= 2 and chat[0]['role'] == 'user' and chat[1]['role'] == 'assistant':
                prompt_in_llama3_format = tokenizer.apply_chat_template(chat[:1], tokenize=False)
                prompt_response_in_llama3_format = tokenizer.apply_chat_template(chat[:2], tokenize=False)
                new_item = {
                    'prompt': prompt_in_llama3_format,
                    'text': prompt_response_in_llama3_format,
                }
                writer.write(new_item)
        writer.close()

    print(f'Finished writing {output_file}')


if __name__ == '__main__':
    data_dir = '/work/avner/data/share_gpt'
    data_file = f'{data_dir}/ShareGPT_V4.3_unfiltered_cleaned_split.json'
    dataset = load_dataset('json', data_files=[data_file], split='train', streaming=False)

    chats = []
    for item in dataset:
        chat = []
        for turn in item['conversations']:
            chat.append(convert_turn_format(turn))
        chats.append(chat)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct')

    random.seed(1234)
    random.shuffle(chats)

    validation_set_size = 1000
    train_chats, valid_chats = [], []
    for chat in chats:
        if len(valid_chats) < validation_set_size and len(chat) >= 2 and chat[0]['role'] == 'user' and chat[1]['role'] == 'assistant':
            valid_chats.append(chat)
        else:
            train_chats.append(chat)

    
    train_output_file = f'{data_dir}/share_gpt_prompt_response_train.jsonl'
    valid_output_file = f'{data_dir}/share_gpt_prompt_response_valid.jsonl'
    write_prompt_response(train_chats, train_output_file)
    write_prompt_response(valid_chats, valid_output_file)

    train_output_file = f'{data_dir}/share_gpt_all_turns_train.jsonl'
    valid_output_file = f'{data_dir}/share_gpt_all_turns_valid.jsonl'
    write_all_turns(train_chats, train_output_file)
    write_all_turns(valid_chats, valid_output_file)
