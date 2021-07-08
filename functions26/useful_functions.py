# 2019-19-11
# This code was made for use in the Fu lab
# by Vasilis Niaouris


import numpy as np
import smtplib  # for emails
import ssl  # for emails
import re

from socket import gaierror  # for email server errors
# from validate_email import validate_email # to check if adress excists


def calculate_weighted_mean(array, weights):
    return np.sum(array*weights)/np.sum(weights)


def get_closest(lst, value):
    lst = np.asarray(lst)
    closest_index = (np.abs(lst - value)).argmin()
    difference_value_minus_list_value = -1
    if value > lst[closest_index]:
        difference_value_minus_list_value = 1
    elif value == lst[closest_index]:
        difference_value_minus_list_value = 0

    return closest_index, difference_value_minus_list_value


def get_added_label_from_unit(astu):
    if astu[0] != '_':
        raise RuntimeError('added_string_to_unit does not begin with: _')

    list_of_non_units = []
    label_string = ''
    added_string_breakdown = re.split("_per_|_times_|_", astu)
    print(added_string_breakdown[1:])
    for added_unit in added_string_breakdown[1:]:
        print(added_unit)
        start_index = astu.find(added_unit)
        if astu[start_index - 5:start_index] == '_per_':
            label_string += '/' + added_unit
        elif astu[start_index - 7:start_index] == '_times_':
            label_string += '*' + added_unit
        else:
            list_of_non_units.append(added_unit)

    for non_units in list_of_non_units:
        label_string += ' ' + non_units.capitalize() + ' '

    return label_string


def send_email(sender_id='', sender_server='gmail.com', sender_password='', recipients='', subject='', message=''):
    # The following code is a modified version of an example found on https://realpython.com/python-send-email/ and
    # https://julien.danjou.info/sending-emails-in-python-tutorial-code-examples/ and for mail validation, the pyPI page

    if len(recipients) == 0:
        print('No recipients were given. Failed to send e-mail with message: ' + message)
        return 5

    # setting up server and sender parameters
    port = 587  # For starttls
    smtp_server = 'smtp.' + sender_server
    sender_email = sender_id + r'@' + sender_server

    # setting up email content
    context = ssl.create_default_context()
    # Adding the subject, and showing the sender (not necessary), and the other recipients (necessary)
    message = 'Subject: {}\nFrom: {}\nTo: {}\n\n{}'.format(subject, sender_email, ','.join(recipients), message)

    # checking recipient validity
    # recipients, invalid_recipients, error = valid_emails(recipients)

    # trying to send email
    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.ehlo()  # Can be omitted
            server.starttls(context=context)
            server.ehlo()  # Can be omitted
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, message)
            server.quit()
    except (gaierror, ConnectionRefusedError):
        # tell the script to report if your message was sent or which errors need to be fixed
        print('Failed to connect to the server. Bad connection settings?')
        return 1
    except smtplib.SMTPServerDisconnected:
        print('Failed to connect to the server. Wrong user/password?')
        return 2
    except smtplib.SMTPException as e:
        print('SMTP error occurred: ' + str(e))
        return 3
    else:
        print('Sent e-mail with message:\n' + message + '\n\nto: ' + ', '.join(recipients))
        return 0


def add_extension_if_necessary(filename, extension):
    if extension[0] != '.':
        extension = '.' + extension
    # finding if filename has ending same as extension and if not, it adds the extension ending to the file
    filename_ending = filename.split('.')[-1]
    if filename_ending != extension[1:]:
        filename = filename + extension

    return filename


def is_str_containing_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_iterable(object):
    try:
        iterator = iter(object)
        return True
    except TypeError:
        return False


# def valid_emails(recipients):
#     # checking recipient validity (only if the email is valid, not if it actually exists).
#     if len(recipients) == 0:
#         print('No recipients were given')
#         return 5
#
#     invalid_recipients = []
#     if isinstance(recipients,list):
#         for i in range(len(recipients)):
#             if not validate_email(recipients[i]):
#                 invalid_recipients = recipients.pop(i)
#     else:
#         if not validate_email(recipients):
#             invalid_recipients = recipients
#             recipients = ''
#
#     if len(invalid_recipients) > 0 and len(recipients) > 0:
#         print('The following recipient addresses are invalid: ' + str(invalid_recipients))
#     elif len(recipients) == 0:
#         print('All recipients are invalid: ' + str(invalid_recipients))
#         return recipients, invalid_recipients, 4
#     return recipients, invalid_recipients, 0

