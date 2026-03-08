with open('anchor_eval.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Force utf-8 on all file opens
content = content.replace("open(jsonl_path, 'a')", "open(jsonl_path, 'a', encoding='utf-8')")
content = content.replace("open(txt_path, 'a')", "open(txt_path, 'a', encoding='utf-8')")
content = content.replace("open(state_path) ", "open(state_path, encoding='utf-8') ")
content = content.replace("open(state_path, 'w')", "open(state_path, 'w', encoding='utf-8')")

# Replace unicode symbols with ASCII
content = content.replace('\u2248 flat', '~ flat')
content = content.replace('\u2191 better', '> better')
content = content.replace('\u2193 worse', '< worse')
content = content.replace('\u2248', '~')
content = content.replace('\u2191', '>')
content = content.replace('\u2193', '<')
content = content.replace('\u0394', 'd')
content = content.replace('\u2013', '-')

with open('anchor_eval.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
