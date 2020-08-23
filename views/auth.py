import json
import random
import hashlib
from time import time
from bson.objectid import ObjectId
import settings
from aiohttp import web
from aiohttp_session import get_session
from aiohttp_session.cookie_storage import EncryptedCookieStorage

import server_data

def check_user(login, password):
    result = False
    if password:
        # print(login, ADMIN_USER, hashlib.md5(password.encode('utf-8')).hexdigest(), ADMIN_PASSWORD)
        if hashlib.md5(password.encode('utf-8')).hexdigest() == settings.ADMIN_PASSWORD and login == settings.ADMIN_USER:
            result = True
    return result

def set_session(session, user_id, request):
    session['user'] = str(user_id)
    session['last_visit'] = time()
    hash = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890") for _ in range(16))
    session['hash'] = hash
    server_data.session_list.append(hash)
    print("set new session", user_id, session)
    raise web.HTTPFound('/static/index.html')

def convert_json(message):
    return json.dumps({'error': message})

class Sign(web.View):
    async def post(self):
        # print("Login start post")
        action = self.request.match_info.get('action', '1')
        data = await self.request.post()
        session = await get_session(self.request)
        if action == '0':
            if data:
                login = data.get('login')
                password = data.get('password')
                result = check_user(login, password)
                if result:
                    set_session(session, login, self.request)
                else:
                    print("Login can not find user")
                    if session.get("user"):
                        del session['user']
                    raise web.HTTPFound("/static/sign.html")
            else:
                raise web.HTTPFound("/static/sign.html")
        else:
            hash = session.get('hash')
            if hash:
                del session['hash']
                if hash in server_data.session_list:
                    server_data.session_list.remove(hash)
            if session.get("user"): 
                del session['user']
            raise web.HTTPFound('/static/login.html')