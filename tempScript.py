import os

source_dir = 'Images'

if not os.path.exists(source_dir):
    print(f"Ошибка: Директория {source_dir} не существует.")
    exit(1)

folder_names = []

for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    if os.path.isdir(folder_path):
        folder_names.append(folder)

sorted_list = sorted(folder_names)

formatted_lines = [f"{index}: '{line}'," for index, line in enumerate(sorted_list)]

with open('result.txt', 'w', encoding='utf-8') as file:
    for formatted_line in formatted_lines:
        file.write(f"{formatted_line}\n")