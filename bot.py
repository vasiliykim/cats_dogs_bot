# coding=utf-8

import cv2
import requests
import tensorflow as tf

import traceback
import sys

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram.keyboardbutton import KeyboardButton
from telegram.replykeyboardmarkup import ReplyKeyboardMarkup
from telegram.replykeyboardremove import ReplyKeyboardRemove

CATEGORIES = ["Dogs", "Cats"]
model = tf.keras.models.load_model("network.model")


# commands
def start(bot, update):
    # user_id = update.message.chat.id
    update.message.reply_text('You are welcome!')


def help(bot, update):
    update.message.reply_text(
        'I can recognize cats and dogs. Send me image of your pet. Current tested accuracy is 86%')


def author(bot, update):
    update.message.reply_text('https://github.com/vasiliykim98')


def echo(bot, update):
    update.message.reply_text('I work only with images')


def image(bot, update):
    image_width = 81
    image_height = 81
    img_name = "image_from_user.jpg"
    try:
        photo_id = update.message['photo'][-1]['file_id']
        img_from_user = bot.getFile(photo_id)

        img_from_user.download(img_name)

        loaded_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        prepared_img = cv2.resize(loaded_img, (image_width, image_height))
        prepared_img = prepared_img.reshape(-1, image_width, image_height, 1)
        prepared_img = prepared_img / 255.0
        prediction = model.predict([prepared_img])
        response = CATEGORIES[int(round(prediction[0][0]))]
        update.message.reply_text(response)
    except Exception:
        print(traceback.format_exc())
        update.message.reply_text("There are some problem with the image")


# keyboards
def get_menu(bot, update):
    kb = [[KeyboardButton('/help')],
          [KeyboardButton('/back (назад)')]]
    reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True)

    update.message.reply_text('My commands', reply_markup=reply_markup)


def get_back(bot, update):
    reply_markup = ReplyKeyboardRemove()
    update.message.reply_text('Send me images', reply_markup=reply_markup)


def main():
    token = ""
    api_url = "https://api.telegram.org/bot{}/".format(token)
    method = 'getUpdates'
    params = {'timeout': 10, 'offset': None}
    try:
        resp = requests.get(api_url + method, params)
        print(resp)
        try:

            updater = Updater(token)

            dp = updater.dispatcher

            dp.add_handler(CommandHandler("start", start))
            dp.add_handler(CommandHandler("help", help))
            dp.add_handler(CommandHandler("author", author))
            dp.add_handler(CommandHandler("menu", get_menu))
            dp.add_handler(CommandHandler("back", get_back))

            dp.add_handler(MessageHandler(Filters.text, echo))

            dp.add_handler(MessageHandler(Filters.photo, image))

            # Start the Bot
            updater.start_polling(poll_interval=0.1, timeout=20)

            print(updater.is_idle)
            updater.idle()



        except Exception:
            sys.exit()

    except Exception:
        sys.exit()


import os

if __name__ == '__main__':
    main()
