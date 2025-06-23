import argparse
import os
from os import system
import smtplib
from email.mime.text import MIMEText
from email.header import Header


def email(count):
    # 邮件内容
    subject = '迭代完成'
    body = "完成！共 " + str(count) + " 轮！"

    # 构建邮件
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = '2337342726@qq.com'
    msg['To'] = '23S103195@stu.hit.edu.cn'
    # msg['To'] = '2455701175@qq.com'

    # 发送邮件
    smtp_server = 'smtp.qq.com'
    smtp_port = 587
    sender_email = '2337342726@qq.com'
    password = 'wmfenccuztioeajh'  # 在QQ邮箱设置里拿到的码

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, [msg['To']], msg.as_string())
        print('邮件发送成功')
    except smtplib.SMTPException as e:
        print('邮件发送失败:', str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input loop and interval')
    parser.add_argument('--count', default=1, help='循环次数')
    parser.add_argument('--interval', default=30, help='间隔时间')
    args = parser.parse_args()
    # 函数操作
    # 发送邮件
    email(args.count)
    # 关机
    # system("shutdown -s -t 60")