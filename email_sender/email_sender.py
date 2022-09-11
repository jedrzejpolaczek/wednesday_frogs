import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


def send_email(UserName, UserPassword, ImgFileName):
    with open(ImgFileName, 'rb') as f:
        img_data = f.read()

    msg = MIMEMultipart()
    msg['Subject'] = 'subject'
    msg['From'] = 'jedrzej.polaczek@mail.com'
    msg['To'] = 'elzeijp@gmail.com'

    text = MIMEText("test")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 465)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(UserName, UserPassword)
    s.sendmail(msg['From'], msg['To'], msg.as_string())
    s.quit()


def is_it_wednesday():
    # If today is Wednesday (0 = Mon, 1 = Tue, 2 = Wen ...)
    return True if datetime.today().weekday() == 2 else False


def send_email_on_wednesday(UserName, UserPassword, ImgFileName):
    if is_it_wednesday:
        # TODO: load model
        # TODO: generate image

        send_email(UserName, UserPassword, ImgFileName)
        