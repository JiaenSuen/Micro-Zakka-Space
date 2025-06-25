import os
import xml.etree.ElementTree as ET

def convert_voc_to_haar_format(folder_path):
    files = os.listdir(folder_path)
    xml_files = [f for f in files if f.endswith('.xml')]

    for xml_file in xml_files:
        xml_path = os.path.join(folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 對應的圖片檔名
        image_filename = root.find('filename').text

        objects = root.findall('object')
        line = f"{len(objects)}"

        for obj in objects:
            bbox = obj.find('bndbox')
            x = int(bbox.find('xmin').text)
            y = int(bbox.find('ymin').text)
            w = int(bbox.find('xmax').text) - x
            h = int(bbox.find('ymax').text) - y
            line += f" {x} {y} {w} {h}"

        # 輸出為與 XML 同名的 txt 檔（取代 XML）
        txt_filename = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(folder_path, txt_filename)

        with open(txt_path, 'w') as f:
            f.write(line + '\n')

        print(f" Converted: {xml_file} -> {txt_filename}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python convert_form.py <標註與圖片的資料夾路徑>")
    else:
        convert_voc_to_haar_format(sys.argv[1])
