from datetime import datetime
import json
from xmlrpc.client import Boolean
from loguru import logger
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def send_email() -> None:
    """ Send email with image. """
    emails_data = get_email_data()
    
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


def get_email_data() -> dict:
    """ 
    Read JSON dict from file.
    
    return dict: dict based on read JSON file.
    """
    logger.info("Opening JSON file.")
    json_file = open('config.json')
    
    logger.info("Returns JSON object as a dictionary.")
    email_data = json.load(json_file)

    return email_data


def send_email_on_wednesday() -> None:
    """ Send email with generated image on each Wednesday. """
    if is_it_wednesday:
        logger.info("It is Wednesday!")
        logger.info("Loading model.")
        # TODO: load model

        logger.info("Generating new frog.")
        # TODO: generate image

        logger.info("Beginning procedure of sending emails.")
        send_email()
    else:
        logger.info("It is not Wednesday.")
        logger.info(" :( ")
        