#TODO
# сервер запускает менеджер ресурсов (GPUs and CPU)
# менеджер ресурсов запускает менеджер потоков на которых уже крутятяся разные видеостримы
#! /usr/bin/env python
import asyncio
# import aiohttp_debugtoolbar
from aiohttp_session import session_middleware
# from aiohttp_session.cookie_storage import EncryptedCookieStorage

from aiohttp import web, WSMsgType
from typing import Any, AsyncIterator, Awaitable, Callable, Dict
import aiosqlite
from pathlib import Path
# from routes import routes
from middlewares import authorize
# from motor import motor_asyncio as ma
# import asyncio
import io, sys
import sqlite3
import cv2 
import json
import os
import settings
import hashlib
import ssl
from views.websocket import WebSocket
import logging
import yaml
import time

#import camera
import gpusManager

log = logging.getLogger('app')
log.setLevel(logging.DEBUG)
f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(f)
log.addHandler(ch)
camerasListData = {'cameras': [['id','name','online','counting','comments','url','borders'],['2','name2','online2','counting2','comments2','url2','borders2']]}

async def camerasList(request):
    data = {'cameras': [['id','name','online','counting','comments','url','borders'],['2','name2','online2','counting2','comments2','url2','borders2']]}
    return  web.json_response(data)

async def filesList(request):
    #params = request.rel_url.query
    #print(params)
    #print(params['file_1'])
    #// json data structure:  name, size in GB, last change time
    data = {'files':{'39.avi':[0.4,'20.10.2020'], 'new':{'45.avi':[0.4,'20.10.2020']}}};
    return  web.json_response(data)

async def addToQueue(request):
    msg_json = json.loads('data from post')

    return  web.json_response({'answer':['addToQueue','ok',request]})

async def getFileImg(request):
    try:
        print("start getFileImag")
        if ('file' in request.rel_url.query):
            fileName = request.rel_url.query['file']
            print("filaName",fileName) #'video/39.avi'            
            if os.path.isfile(fileName):
                vidcap = cv2.VideoCapture(fileName)
                img = vidcap.read()[1]
                img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)
                res = cv2.imencode('.JPEG', img)[1].tobytes()
                #return web.Response(text="All right!")
                return web.Response(body=res)
            else:
                print(fileName,"can not find file")
                return web.json_response({'error':["can not find file", fileName]})
        else:
            return web.json_response({'error':["can not find file name"]})
    except:
        print("can not open file")
        await web.json_response({'error':[fileName,"can not open file"]})

async def Index(request):
    return web.HTTPFound('static/testVideo.html')

async def WebSocketCmd(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app['websocketscmd'].add(ws)
    
    #totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                print("data=", type(msg.data), msg.data)
                msg_json = json.loads(msg.data)
                print("json=", type(msg_json), msg_json['cmd'])
                if msg_json['cmd'] == 'close':
                    await ws.close()
                elif msg_json['cmd'] == 'getCameras':
                    # print(camerasListData)
                    await ws.send_json(camerasListData)
                elif msg_json['cmd'] == 'getFileImag':
                    print("testImg")
                    img = cv2.imread('video/cars.jpg')
                    print("img", len(img))
                    res = cv2.imencode('.jpg', img)[1].tobytes()
                    print("res", len(res))
                    cv2.imwrite("video/dd.jpg", img)
                    await ws.send_bytes(res)
                else:
                    print("error websocket command")
                    await ws.send_json({'error':{'unknown', msg.data}})
            except:
                print(sys.exc_info())
                await ws.send_json({'error':["can not parse json", msg.data]})
        elif msg.type == WSMsgType.ERROR:
            print('ws connection closed with exception %s' % ws.exception())
    print('websocket connection closed')
    request.app['websocketscmd'].remove(ws)
    return ws

routes = [
    ('GET', '/',  Index),
    ('GET', '/ws',  WebSocket),
    ('GET', '/wsCmd',  WebSocketCmd),
    #('GET', '/testImg',  testImg),
    # ('*',   '/login',   Login,     'login'),
    # ('POST', '/sign/{action}',  Sign,    'sign'),
    ('GET',  '/camerasList', camerasList),
    ('GET',  '/getFileImg', getFileImg),
    ('GET',  '/filesList', filesList),
    ('POST',  '/addToQueue', addToQueue),

]

async def on_shutdown(app):
    for ws in app['websockets']:
        await ws.close(code=1001, message='Server shutdown')
    for ws in app['websocketscmd']:
        await ws.close(code=1001, message='Server shutdown')
# print("SECRET_KEY", SECRET_KEY)
#middle = [
#    session_middleware(EncryptedCookieStorage(hashlib.sha256(bytes(settings.SECRET_KEY, 'utf-8')).digest())),
#    authorize,
#]

#app = web.Application(middlewares=middle)
app = web.Application()

# route part
for route in routes:
    app.router.add_route(route[0], route[1], route[2])
app['static_root_url'] = '/static'
app.router.add_static('/static', 'static', name='static', append_version=True)
# app.router.add_static('/', 'index', name='static')
# end route part

#  background task 
async def background_process():
    while True:
        log.debug('Run background task each 1 min')
        print("len websocketscmd:", str(len(app['websocketscmd'])))
        try:
            if 'manager' in app:
                if any(app['manager'].gpusActiveList):
                    print("cnt=", app['manager'].gpusActiveList['test'].cnt)
                    #app['manager'].camActiveObjList['test'].cnt = 20

        except:
            print('Errror')
            print(sys.exc_info())
        if len(app['websocketscmd'])>0:
            print("start send back")
            try:
                for client in app['websocketscmd']:
                    await client.send_json({'camera':[1,2,3]})
            except:
                print("Unexpected error:", sys.exc_info()[0])
        else:
            print("len=0")
        await asyncio.sleep(6)

#async def start_background_tasks(app):
#    app['dispatch'] = asyncio.create_task(background_process())

#async def cleanup_background_tasks(app):
#    app['dispatch'].cancel()
#    await app['dispatch']
#app.on_startup.append(start_background_tasks)
#app.on_cleanup.append(cleanup_background_tasks)
#  end background task

def get_db_path():
    here = Path.cwd()/ settings.DB_PATH
    return here 

# db connect
async def init_db(app: web.Application):
    sqlite_db = get_db_path()
    db = await aiosqlite.connect(sqlite_db)
    db.row_factory = aiosqlite.Row
    app.db = db
    yield
    await db.close()
app.cleanup_ctx.append(init_db)

# end db connect

def try_make_db():
    log.debug("DB path: " + str(settings.DB_PATH))
    sqlite_db = get_db_path()
    if sqlite_db.exists():
        log.debug("DB exist")
        return
    log.debug("creating new DB")
    with sqlite3.connect(sqlite_db) as conn:
        cur = conn.cursor()

        # cur.execute(f'CREATE TABLE {MESSAGE_COLLECTION} (id INTEGER PRIMARY KEY, title TEXT, text TEXT, owner TEXT, editor TEXT, image BLOB)')
        # conn.commit()

        sql = f'CREATE TABLE {settings.USER_COLLECTION} (id INTEGER PRIMARY KEY, login TEXT, email TEXT, password TEXT)'
        print(sql)
        #log.debug("sql:"+sql)
        cur.execute(sql)
        conn.commit()

try_make_db()

app.on_cleanup.append(on_shutdown)
app['websocketscmd'] = set()
#  start cameras manager Object
print("starting GPU manager")
# app['manager'] = set()
# app['manager'] = gpusManager.Manager("Gpus_manager_default")
# time.sleep(10)

# manager.daemon = True
log.info('Running...')
web.run_app(app)
#  Stop cameras manager Object
if 'manager' in app:
    app['manager'].kill()
log.info('The server stopped!')
