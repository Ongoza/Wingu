#TODO
# сервер запускает менеджер ресурсов (GPUs and CPU)
# менеджер ресурсов запускает менеджер потоков на которых уже крутятяся разные видеостримы
#! /usr/bin/env python
import asyncio
# import aiohttp_debugtoolbar
from aiohttp_session import session_middleware
# from aiohttp_session.cookie_storage import EncryptedCookieStorage
import numpy as np
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
managerConfigFile = "default" 
log = logging.getLogger('app')
log.setLevel(logging.DEBUG)
f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(f)
log.addHandler(ch)
camerasListData = {'cameras': [['id','name','online','counting','comments','url','borders'],['2','name2','online2','counting2','comments2','url2','borders2']]}

test_manager_config = {'managerConfig': {'cpu_config': 0, 'autostart_gpus_list': None, 'gpus_configs_list': [0, 1], 'autotart_streams': None, 'streams': [39, 43], 'gpu_configs': {}, 'streams_configs': {}}, 
                       'gpusConfigList': {'cpu': {'id': 'gpu_0', 'batch_size': 32, 'img_size': 416, 'detector_name': 'yolov4', 'detector_filename': 'yolov4.h5', 'yolo_max_boxes': 100, 'yolo_iou_threshold': 0.5, 'yolo_score_threshold': 0.5}}, 
                       }

test_streams_config =  {'streamsConfigList': 
                          {
                           '39': {'id': 39, 'name': 'test00', 'uid': 'lynE3ce', 'url': 'video/39.avi', 'isFromFile': True, 'save_path': 'video/39_out.avi', 'body_min_w': 64, 'path_track': 20, 'body_res': [256, 128], 'display_video_flag': True, 'max_cosine_distance': 0.2, 'save_video_flag': True, 'skip_frames': 0, 'encoder_filename': 'mars-small128.pb', 'batch_size': 32, 'img_size_start': [1600, 1200], 'save_video_res': [720, 540], 'borders': {'border1': [[0, 104], [312, 104]]}}
                            ,'43': {'id': 43, 'name':'test01', 'url': 'video/43.avi', 'isFromFile': True, 'save_path': 'video/43_out.avi', 'body_min_w': 64, 'path_track': 20, 'body_res': [256, 128], 'display_video_flag': True, 'max_cosine_distance': 0.2, 'save_video_flag': True, 'skip_frames': 2, 'encoder_filename': 'mars-small128.pb', 'batch_size': 32, 'img_size_start': [1600, 1200], 'save_video_res': [720, 540], 'borders': {'border1': [[0, 104], [312, 104]]}}}
                       }

#async def camerasList(request):
#    data = {'cameras': [['id','name','online','counting','comments','url','borders'],['2','name2','online2','counting2','comments2','url2','borders2']]}
#    return  web.json_response(data)

#async def getFilesList(request):
#    #params = request.rel_url.query
#    #print(params)
#    #print(params['file_1'])
#    #// json data structure:  name, size in GB, last change time
#    data = {'files':{'39.avi':[0.4,'20.10.2020'], 'new':{'45.avi':[0.4,'20.10.2020']}}};
#    return  web.json_response(data)

#async def addToQueue(request):
#    msg_json = json.loads('data from post')
#    print(msg_json)
#    return  web.json_response({'answer':['addToQueue','ok',request]})

async def getFileImg(request):
    try:
        print("start getFileImag")
        if ('file' in request.rel_url.query):
            fileName = request.rel_url.query['file']
            print("filaName", fileName) #'video/39.avi'            
            if os.path.isfile(fileName):
                vidcap = cv2.VideoCapture(fileName)
                img = vidcap.read()[1]
                img = cv2.resize(img, (541,416), interpolation = cv2.INTER_AREA)
                res = cv2.imencode('.JPEG', img)[1].tobytes()
                #exif_dict = piexif.load(res.info['exif'])
                #print("res", exif_dict)
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
    return web.HTTPFound('static/cameras.html')

async def saveConfig(ws, config):
    res = {'OK':["saveConfig", config['tp'], config['name']]}
    print(res)
    fileName = os.path.join('config', config['tp'] + str(config['name'])+'.yaml')
    try:
        if os.path.isfile(fileName):
            log.info("Create backup for config " + fileName)
            # !!!!!!!!!!!!!!!!!!!!!!!!
        with open(fileName, 'w', encoding='utf-8') as f:    
            yaml.dump(config['data'], f)
        #if manager in app:
        #    app["manager"].updateConfig(config)
    except:
          log.error("Can not save config for " + fileName )
          res = {'error':['saveConfig', config['tp'], config['name']]}
          print(sys.exc_info())
    await ws.send_json(res)

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
                elif msg_json['cmd'] == 'getStreamsConfig':
                    # if "manager" in app:
                        # data = app["manager"].getStreamsConfig()
                    data = test_streams_config                    
                    await ws.send_json(data)
                elif msg_json['cmd'] == 'getManagerData':
                    if True:
                    # if "manager" in app:
                        # data = app["manager"].getConfig()
                        # print("test_manager_config")
                        data = test_manager_config
                        # print("data", data)
                        await ws.send_json(data)
                    else:
                        print(sys.exc_info())
                        await ws.send_json({'error':['getManagerData', msg.data]})
                elif msg_json['cmd'] == 'saveStream':
                    print("saveStream", msg_json['config'])                    
                    await saveConfig(ws, msg_json['config'])
                elif msg_json['cmd'] == 'startStream':
                    print("startStream", msg_json['stream_id'])
                    await ws.send_json({'OK':["startStream", msg_json['stream_id']]})
                elif msg_json['cmd'] == 'startGetStream':
                    print("startGetStream", msg_json['stream_id'])

                    await ws.send_json({'OK':["startGetStream", msg_json['stream_id']]})
                #elif msg_json['cmd'] == 'getCameras':
                #    # print(camerasListData)
                #    await ws.send_json(camerasListData)
                #elif msg_json['cmd'] == 'getFileImg':
                #    print("testImg")
                #    #arr_symb = np.fromstring('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', dtype=np.uint8)
                #    #uid_g = np.append(np.random.choice(arr_symb, 7), np.uint8(1))
                #    img = cv2.imread('video/cars.jpg')
                #    res = cv2.imencode('.jpg', img)[1]
                #    uid_g = np.array([78, 55, 68, 79, 114, 97, 114, 1], dtype=np.uint8)
                #    # N7DOrar
                #    res = np.append(res, uid_g)
                #    #data = np.array([1,2], dtype=np.uint8)
                #    #res = np.concatenate(res, data)
                #    # print("res",type(res), res.dtype, res.size, uid_g, res[-8:])
                #    # cv2.imwrite("video/dd.jpg", img)
                #    await ws.send_bytes(res.tobytes())
                else:
                    print("error websocket command")
                    await ws.send_json({'error':['unknown', msg.data]})
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
    #('GET',  '/camerasList', camerasList),
    ('GET',  '/getFileImg', getFileImg),
    #('GET',  '/filesList', getFilesList),
    #('POST',  '/addToQueue', addToQueue),
    #('POST',  '/saveConfig', saveConfig),

]

async def on_shutdown(app):
    try:
        if  'websockets' in app:
            for ws in app['websockets']:
                await ws.close(code=1001, message='Server shutdown')
        if  'websocketscmd' in app:
            for ws in app['websocketscmd']:
                await ws.close(code=1001, message='Server shutdown')
    except:
        print(sys.exc_info())
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
#app['manager'] = gpusManager.Manager(managerConfigFile)
#time.sleep(10)

# manager.daemon = True
log.info('Running...')
web.run_app(app)
#  Stop cameras manager Object
if 'manager' in app:
    app['manager'].kill()
log.info('The server stopped!')
# print(sys.exc_info())
