from aiohttp import web
from aiohttp.web import middleware
from aiohttp_session import get_session

import server_data
import settings

def checkAuth(user, hash):
    result = False
    print("checkAuth", user, hash, server_data.session_list)
    if hash in server_data.session_list:
        result = True
    return result

@middleware
async def authorize(request, handler):
    print("start middleware", request.path) 
    if request.path.startswith("/static/lib"):
        return await handler(request)
    elif request.path.startswith("/static/"):
        session = await get_session(request)
        # print(" auth ok 1", request.path, session.get('user'))
        if checkAuth(session.get('user'), session.get('hash')):
            print("auth ok 2", session.get('user'))
            return await handler(request)
        else:
            if request.path.startswith('/static/sign.html'):
                # print("sign ok", request)
                responce = await handler(request)
                print("sign responce", responce)
                return responce
            else:
                # print("redirect to login page", request.path)
                raise web.HTTPFound('/static/sign.html')
    elif request.path.startswith("/sign"):
        responce = await handler(request)
        print("post responce", responce)
        return responce
    else:
        if request.path == '/' or request.path == '':
            print("mid redirect", request.path)
            session = await get_session(request)
            if checkAuth(session.get('user'), session.get('hash')):
                raise web.HTTPFound('/static/index.html')
            else:
                raise web.HTTPFound('/static/sign.html')
        print("mid error", request.path)
        raise web.HTTPForbidden(body=b'Forbidden')
        # return web.HTTPFound('/static/sign.html')