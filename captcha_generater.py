from captcha.image import ImageCaptcha
import random, os
import config

def gen_special_img(text, width, height, file_path):
    generator = ImageCaptcha(width=width, height=height)
    img = generator.generate_image(text)
    img.save(file_path, format = 'PNG')


captcha_train = './samples/generate/train'
captcha_test = './samples/generate/test'


train_count = 200000
test_count = 1000

characters = config.CHAR_VECTOR
# for count in range(0, train_count):
#     text = ''
#     for i in range(4):
#         text += random.choice(characters)
#     save_path = os.path.join(captcha_train, '{}_{}.png'.format(text, count))
#     gen_special_img(text, 160, 70, save_path)
#     print(text)

for test_count in range(0, test_count):
    text = ''
    for i in range(4):
        text += random.choice(characters)
    save_path = os.path.join(captcha_test, '{}_{}.png'.format(text, test_count))
    gen_special_img(text, 160, 70, save_path)