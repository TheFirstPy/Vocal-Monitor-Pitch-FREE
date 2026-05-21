"""Создание логотипа с прозрачным фоном для Vocal Pitch Monitor"""
from PIL import Image, ImageDraw, ImageFont

# Создаем изображение с прозрачным фоном
size = 256
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Градиентный круг (микрофон)
center = size // 2
radius = size // 2 - 16

# Рисуем внешний круг (корпус микрофона)
for i in range(radius, radius - 20, -1):
    alpha = int(255 * (1 - (radius - i) / 20))
    color = (79, 195, 247, alpha)  # C_SONG цвет
    draw.ellipse([center - i, center - i, center + i, center + i], fill=color)

# Внутренний круг (сетка микрофона)
inner_radius = radius - 25
for i in range(inner_radius, inner_radius - 15, -1):
    alpha = int(255 * (1 - (inner_radius - i) / 15))
    color = (255, 202, 40, alpha)  # C_USER цвет
    draw.ellipse([center - i, center - i, center + i, center + i], fill=color)

# Добавляем ноту внутри
try:
    font = ImageFont.truetype("arial.ttf", 80)
except:
    font = ImageFont.load_default()

# Рисуем музыкальную ноту
note_color = (255, 255, 255, 255)
draw.text((center - 30, center - 50), "♪", fill=note_color, font=font)

# Ножка микрофона внизу
draw.rectangle([center - 8, center + radius - 30, center + 8, center + radius], 
               fill=(100, 100, 120, 200))

# Сохраняем как ICO с прозрачностью
img.save('/workspace/assets/logo.ico', format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
img.save('/workspace/assets/logo.png', format='PNG')

print("Логотип создан: assets/logo.ico и assets/logo.png")
