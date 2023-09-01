import random
import string

def generate_random_data():
    image_paths = []
    for _ in range(100):
        # 随机生成图像路径，使用8个随机字符
        random_path = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        image_path = f"{random_path}.jpg"
        image_paths.append(image_path)

    data = []
    for image_path in image_paths:
        # 随机生成四个角点的像素坐标值
        x1 = random.randint(0, 100)
        y1 = random.randint(0, 100)
        x2 = random.randint(0, 100)
        y2 = random.randint(0, 100)
        x3 = random.randint(0, 100)
        y3 = random.randint(0, 100)
        x4 = random.randint(0, 100)
        y4 = random.randint(0, 100)
        # 将数据格式化为字符串
        line = f"{image_path},{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4}"
        data.append(line)

    return data

# 生成随机数据
random_data = generate_random_data()

# 将数据写入txt文件
with open("data.txt", "w") as file:
    file.write("\n".join(random_data))