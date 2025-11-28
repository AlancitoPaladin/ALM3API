import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_password(email, password):
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()

    smtp.login('edutecteam@gmail.com', 'zhev jmly rpqi aisb')

    def message(subject="Password Recovery", text=" "):
        msg = MIMEMultipart()

        msg['Subject'] = subject

        msg.attach(MIMEText(text))
        return msg

    msg = message("Reseteo de contraseña", f"""Si esta recibiendo este correo, se debe a que usted solicito que se cambie su contraseña.
    S
    u nueva contraseña se muestra a continuacion: {password}

     Atte. EduTecTeam Support""")

    smtp.sendmail(from_addr="edutecteam@gmail.com",
                  to_addrs=email, msg=msg.as_string())

    smtp.quit()
