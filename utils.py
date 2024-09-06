import smtplib, ssl

def send_email(prob):
    port = 465
    smtp_server = "smtp.odido.com"
    sender_email = "mlmodel@odido.com"
    receiver_email = "stakeholder@odido.com"
    password = "****"
    message = f"""\
    Dear Stakeholder,

    Please find attached the weekly report on product purchase probabilities for our subscribers.
    The CSV file contains the probabilities for each product, sorted by the likelihood of purchasing product_02.

    Probabilities that product02 get sold: {prob}
    """

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
            print("Email sent successfully.")
    except Exception as e:
        print("Email not sent.\n")
        print("Email Content:")
        print(message)