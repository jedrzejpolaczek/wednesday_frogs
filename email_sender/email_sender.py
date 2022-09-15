from datetime import datetime
from xmlrpc.client import Boolean
from loguru import logger
import tensorflow
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import numpy as np
import os

from utils.configuration import get_json_data


def send_email() -> None:
    """ Send email with image. """
    emails_data = get_json_data('email_config.json')
    
    user_name = emails_data["user_name"]
    user_password = emails_data["user_password"]
    sender = emails_data["sender"]
    riecievers = emails_data["riecievers"]
    img_file_name = emails_data["img_file_name"]
    img_file_name_path = emails_data["img_file_name_path"]
    subject = emails_data["subject"]
    body = emails_data["body"]
    
    logger.info("Create instance of MIMEMultipart.")
    msg = MIMEMultipart()
        
    logger.info("Storing the senders email address.")
    msg['From'] = sender
        
    logger.info("Storing the subject.")
    msg['Subject'] = subject
        
    logger.info("Create a string to store the body of the mail.")
    body = body
        
    logger.info("Attach the body with the msg instance.")
    msg.attach(MIMEText(body, 'plain'))
        
    logger.info("Open the file to be sent.")
    filename = img_file_name
    attachment = open(img_file_name_path, "rb")
        
    logger.info("Instance of MIMEBase (to change payload).")
    payload = MIMEBase('application', 'octet-stream')
        
    logger.info("Change the payload.")
    payload.set_payload((attachment).read())
        
    logger.info("Encode into base64.")
    encoders.encode_base64(payload)
        
    payload.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        
    logger.info("Attach the instance of MIMEBase to instance message.")
    msg.attach(payload)
        
    logger.info("Creates SMTP session.")
    smtp_session = smtplib.SMTP('smtp.gmail.com', 587)
        
    logger.info("Start TLS for security.")
    smtp_session.starttls()
        
    logger.info("Authentication using generated app password.")
    smtp_session.login(user_name, user_password)
        
    logger.info("Converts the Multipart msg into a string.")
    text = msg.as_string()
        
    for riciever in riecievers:
        logger.info("Storing the receivers email address.") 
        msg['To'] = riciever

        logger.info("Sending the mail to %s" % riciever)
        smtp_session.sendmail(sender, riciever, text)
        
    logger.info("Terminating the session.")
    smtp_session.quit()


def is_it_wednesday() -> Boolean:
    """ 
    Check if it is Wednesday.
    
    return boolean: True if it is, false if not.
    """
    # If today is Wednesday (0 = Mon, 1 = Tue, 2 = Wen ...)
    return True if datetime.today().weekday() == 2 else False


def generate_and_save_image(gan: tensorflow.Model) -> None:
    """
    Generating and saving on disk generated image.
    
    gan (tensorflow.Model): GAN model.
    """
    random_latent_vectors = np.random.normal(size=(20, 32))
    generated_images = gan.predict(random_latent_vectors)
    img = tensorflow.keras.utils.array_to_img(generated_images[0] * 255., scale=False)
    img.save(os.path.join("", 'generated_frog.png'))


def send_email_on_wednesday() -> None:
    """ Send email with generated image on each Wednesday. """
    if is_it_wednesday:
        logger.info("It is Wednesday!")
        logger.info("Loading model.")
        gan = tensorflow.keras.models.load_model('gan_model\gan.h5')

        logger.info("Generating new frog.")
        generate_and_save_image(gan)  # TODO: these should be to seperate functions.

        logger.info("Beginning procedure of sending emails.")
        send_email()
    else:
        logger.info("It is not Wednesday.")
        logger.info(" :( ")
        