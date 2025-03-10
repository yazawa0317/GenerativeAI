from getpass import getpass
import os
import sys

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

docdir = os.path.dirname(os.path.abspath(__file__)) + r'\.streamlit'
yaml_path = os.path.join(docdir, '.config.yaml')


if __name__=="__main__":
    yaml_data = {"credentials":{"usernames":{}}}
    person = {}
    if os.path.exists(yaml_path):
        with open(yaml_path,"r") as f:
            yaml_data = yaml.safe_load(f)
    
    username = input("your name: ")
    password = getpass("your password: ")

    credentials = {
        "usernames": {}  # ここで "usernames" は空の辞書
    }
    credentials["usernames"][username] = {
        "password": password  # プレーンなパスワード
    }
    
    password = stauth.Hasher.hash_passwords(credentials)
    email = input("your email: ")
    
    person = {
        "name":username,
        "email":email,
        "password":credentials['usernames'][username]['password']
    }
    
    yaml_data["credentials"]["usernames"][username] = person
    
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)
        print("write yaml file!")