import smtplib

from loguru import logger
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from utils import get_json_data


def send_email(json_dir: str="email_config.json") -> None:
    """ Send email with image. """
    emails_data = get_json_data(json_dir)
    
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
    smtp_session = smtplib.SMTP('smtp.gmail.com', 587)  # Fixme: magic number
        
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
        