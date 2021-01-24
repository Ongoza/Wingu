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
from shutil import copyfile
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
import weakref

#import camera
import gpusManager

camerasListData = {'cameras': [['id','name','online','counting','comments','url','borders'],['2','name2','online2','counting2','comments2','url2','borders2']]}

test_manager_config = {'managerConfig': {'cpu_config': 'device0', 'autostart_gpus_list': None, 'gpus_configs_list': ['device0', 'device1'], 'autotart_streams': None, 'streams': ['file_39', 'file_43'] }, 
                        'gpusConfigList': {'cpu': {'id': 0, "device_name":"CPU",  'batch_size': 32, 'img_size': 416, 'detector_name': 'yolov4', 'detector_filename': 'yolov4.h5', 'yolo_max_boxes': 100, 'yolo_iou_threshold': 0.5, 'yolo_score_threshold': 0.5}}, 
                        }

test_streams_config =  {'streamsConfigList': 
                            {
                            'file_39': {'id': 'file_39', 'name': 'test00', 'uid': 'lynE3ce', 'url': 'video/39.avi', 'isFromFile': True, 'save_path': 'video/39_out.avi', 'body_min_w': 64, 'path_track': 20, 'body_res': [256, 128], 'display_video_flag': True, 'max_cosine_distance': 0.2, 'save_video_flag': True, 'skip_frames': 0, 'encoder_filename': 'mars-small128.pb', 'batch_size': 32, 'img_size_start': [1600, 1200], 'save_video_res': [720, 540], 'borders': {'border1': [[0, 104], [312, 104]]}}
                            ,'file_43': {'id': 'file_43', 'name':'test01', 'url': 'video/43.avi', 'isFromFile': True, 'save_path': 'video/43_out.avi', 'body_min_w': 64, 'path_track': 20, 'body_res': [256, 128], 'display_video_flag': True, 'max_cosine_distance': 0.2, 'save_video_flag': True, 'skip_frames': 2, 'encoder_filename': 'mars-small128.pb', 'batch_size': 32, 'img_size_start': [1600, 1200], 'save_video_res': [720, 540], 'borders': {'border1': [[0, 104], [312, 104]]}}}
                        }

counter = 0

############################################################
def get_db_path():
    return "db.wingu.sqlite3"

def init_db():
    try:
        sqlite_db = get_db_path()
        if os.path.isfile(sqlite_db):
            print("DB exist ", counter)
            return
        conn = sqlite3.connect(sqlite_db)
        c = conn.cursor()
        c.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, login TEXT, email TEXT, password TEXT, time INTEGER)')
        c.execute('CREATE TABLE stats (id INTEGER PRIMARY KEY, device TEXT, cpu INTEGER, mem INTEGER, temp INTEGER, time INTEGER)')
        c.execute('CREATE TABLE intersetions (id INTEGER PRIMARY KEY, border TEXT, stream_id TEXT, in_out INTEGER, time INTEGER)')
        #c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")
        conn.commit()
        conn.close()
        print("DB is ok ", counter)
    except:
        print("db is not ok")

async def save_statistic(type_data, id, data):
    t = int(time.time())
    print("data=", type_data, id, data, t)
    sqlite_db = get_db_path()
    try:
        for item in data:
            sql = f'INSERT INTO {type_data}(border, stream_id, time) VALUES("{item}", "{id}", {int(time.time())})'
            print("sql", sql)
            async with aiosqlite.connect(sqlite_db) as db:
                await db.execute(sql)
                await db.commit()

        async with aiosqlite.connect(sqlite_db) as db:
            async with db.execute('SELECT * FROM intersetions') as cursor:
                rows = await cursor.fetchall()
                print(rows)
            # await db.execute("SELECT COUNT(*) FROM intersetions");
    except:
        print("save data")
        print(sys.exc_info())

async def ws_send_data(client, data, binary=False):
    try:
        print("data=", len(data), client)
        if binary:
            await client.send_bytes(data)
        else:
            await client.send_json(data)
        return True
    except:
        print("ws_send_data error")
        print(sys.exc_info())
        try:
            await client.send_json({"error":["can not send data"]})
        except:
            print("ws_send_data error in except")



        # await ws.send_json({"error":['saveConfig', config['tp'], config['name']]})

class Server:
    def __init__(self):
        init_db()
        self.managerConfigFile = "default"
        self.stop = 5
        self.log = logging.getLogger('app')
        self.log.setLevel(logging.DEBUG)
        f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(f)
        self.log.addHandler(ch)
        self.log.addHandler(ch)
        self.routes = [
                ('GET', '/',  self.Index),
                ('GET', '/ws',  WebSocket),
                ('GET', '/wsCmd',  self.WebSocketCmd),
                #('GET', '/testImg',  self.testImg),
                # ('*',   '/login',   Login,     'login'),
                # ('POST', '/sign/{action}',  Sign,    'sign'),
                #('GET',  '/camerasList', camerasList),
                ('GET',  '/getFileImg', self.getFileImg),
                #('GET',  '/filesList', self.getFilesList),
                ('GET',  '/update', self.update),
                #('POST',  '/saveConfig', self.saveConfig),

            ]
        # print("SECRET_KEY", SECRET_KEY)
        #middle = [
        #    session_middleware(EncryptedCookieStorage(hashlib.sha256(bytes(settings.SECRET_KEY, 'utf-8')).digest())),
        #    authorize,
        #]

        #app = web.Application(middlewares=middle)

        # route part
        self.app = web.Application()
        for route in self.routes:
            self.app.router.add_route(route[0], route[1], route[2])
        self.app['static_root_url'] = '/static'
        self.app.router.add_static('/static', 'static', name='static', append_version=True)
        # app.router.add_static('/', 'index', name='static')
        # end route part        
        self.app.on_shutdown.append(self.on_shutdown)
        self.app['websocketscmd'] = weakref.WeakSet()
        #  start cameras manager Object
        print("starting GPU manager")
        self.app.on_startup.append(self.start_background_tasks)
        self.app.on_cleanup.append(self.cleanup_background_tasks)
        #self.app.cleanup_ctx.append(self.init_db)
        #time.sleep(2)
        #print("db", self.app)
        #self.app['manager'] = set()
        self.app['manager'] = gpusManager.Manager(self.managerConfigFile)

        # manager.daemon = True
        self.log.info('Running...')
        web.run_app(self.app)

        #  Stop cameras manager Object
        if 'manager' in self.app:
            self.app['manager'].kill()
        self.log.info('The server stopped!')
        # print(sys.exc_info())

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

    async def saveConfig(self, ws, config):
        res = {'OK':["saveConfig", config['tp'], config['name']]}
        print(res)
        fileName = os.path.join('config', config['tp'] + str(config['name'])+'.yaml')
        try:
            if os.path.isfile(fileName):
                copyfile(fileName, fileName+"."+str(time.time()))
                print("Create backup for config " + fileName)
            with open(fileName, 'w', encoding='utf-8') as f:    
                yaml.dump(config['data'], f)
            if 'manager' in self.app:
                await self.app["manager"].addConfig(config['name'], config['tp'], config['data'], client=ws)
        except:
            self.log.error("Can not save config for " + fileName )
            print(sys.exc_info())

    async def Index(self, request):
        return web.HTTPFound('static/cameras.html')

    async def update(self, request):
        params = request.rel_url.query
        print("Server updated", params)
        if params['cmd']=='startManager':
            print("startManager!!!!")
        elif params['cmd']=='startGPU':
            print("start GPU", params['name'])
        elif params['cmd']=='stopStream':
            print("start GPU", params['name'])
            if 'websocketscmd' in self.app:
                if 'manager' in self.app:
                    for ws in self.app['websocketscmd']:
                        await ws.send_json({"OK":['stopStream', params['name']]})
        elif params['cmd']=='configUpdated':
            print("start configUpdated", params['type'])
            # broadcast updates
            if 'websocketscmd' in self.app:
                if 'manager' in self.app:
                    for ws in self.app['websocketscmd']:
                        await self.app["manager"].getStreamsConfig(ws)
        elif params['cmd']=="frame":
                try:
                    print("server frame updated")
                    if "manager" in self.app:
                        data = self.app["manager"].getFrame()
                        if data is not None:
                            data = data.tobytes()
                            for client in self.app['websocketscmd']:
                                await client.send_bytes(data)
                except:
                    print(sys.exc_info())
        else:
            print("Server update Unknown caommnd")

    async def WebSocketCmd(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        request.app['websocketscmd'].add(ws)
        await ws.send_json({"OK":["start sever", "event on init connection"] })
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
                         if "manager" in self.app:
                            await self.app["manager"].getStreamsConfig(ws)
                            # data = test_streams_config                    
                            # await ws.send_json(data)
                         else:
                            await ws.send_json({"error":['getStreamsConfig', msg.data]})                             
                    elif msg_json['cmd'] == 'getManagerData':
                        if True:
                            #   if "manager" in app:
                            # data = app["manager"].getConfig()
                            # print("test_manager_config")
                            data = test_manager_config
                            # print("data", data)
                            await ws.send_json(data)
                        else:
                            print(sys.exc_info())
                            await ws.send_json({"error":['getManagerData', msg.data]})
                    elif msg_json['cmd'] == 'saveStream':
                        await self.saveConfig(ws, msg_json['config'])
                    elif msg_json['cmd'] == 'stopGetStream':
                        print("stopGetStream", msg_json['stream_id'])
                        if 'manager' in self.app:
                            try:
                               await self.app['manager'].stopGetStream(ws, msg_json['stream_id'])
                            except:
                                await ws.send_json({"error":["stopStream", msg_json['stream_id'], "exception on server"]})
                        else:
                           await ws.send_json({"error":["stopGetStream", msg_json['stream_id'], "mamanger is not running"]})

                    elif msg_json['cmd'] == 'startGetStream':
                        print("startGetStream", msg_json['stream_id'])
                        if 'manager' in self.app:
                            try:
                               await self.app['manager'].startGetStream(ws, msg_json['stream_id'])
                            except:
                                await ws.send_json({"error":["startStream", msg_json['stream_id'], "exception on server"]})
                        else:
                           await ws.send_json({"error":["startGetStream", msg_json['stream_id'], "mamanger is not running"]})
                    elif msg_json['cmd'] == 'startStream':
                        print("startStream", msg_json['stream_id'])
                        if 'manager' in self.app:
                            try:
                               await self.app['manager'].startStream(msg_json['stream_id'], ws)
                            except:
                                print(sys.exc_info())
                                await ws.send_json({"error":["startStream", msg_json['stream_id'], "exception on server"]})
                        else:
                           await ws.send_json({"error":["startStream", msg_json['stream_id'], "mamanger is not running"]})
                    elif msg_json['cmd'] == 'stopStream':
                        print("stopStream", msg_json['stream_id'])
                        if 'manager' in self.app:
                            try:
                               self.app['manager'].stopStream(msg_json['stream_id'])
                            except:
                                print(sys.exc_info())
                                await ws.send_json({"error":["stopStream", msg_json['stream_id'], "exception on server"]})
                        else:
                           await ws.send_json({"error":["stopStream", msg_json['stream_id'], "mamanger is not running"]})

                        # await ws.send_json({'OK':["startStream", msg_json['stream_id']]})
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
                        await ws.send_json({"error":['unknown', msg.data]})
                except:
                    print(sys.exc_info())
                    await ws.send_json({"error":["can not parse json", msg.data]})
            elif msg.type == WSMsgType.ERROR:
                print('ws connection closed with exception %s' % ws.exception())
        # print('websocket connection closed')
        if 'manager' in self.app:
            try:
                await self.app['manager'].stopGetStream(ws, "all")
            except:
                print("server exceptption on stopGetStream on close connection")
                print(sys.exc_info())
        request.app['websocketscmd'].remove(ws)
        return ws


    async def getFileImg(self, request):
        try:
            print("server getFileImag start==============")
            if ('file' in request.rel_url.query):
                fileName = request.rel_url.query['file']
                print("filaName", fileName) #'video/39.avi'            
                # if os.path.isfile(fileName):
                vidcap = cv2.VideoCapture(fileName)
                time.sleep(1)
                if vidcap.isOpened():
                    print("server getFileImag ok==============")
                    ret, img = vidcap.read()
                    img = cv2.resize(img, (541,416), interpolation=cv2.INTER_AREA)
                    res = cv2.imencode('.JPEG', img)[1].tobytes()
                    vidcap.release()
                    return web.Response(body=res)
                else:
                    print("server getFileImag wait more==============")
                    time.sleep(3)
                    if vidcap.isOpened():
                        ret, img = vidcap.read()
                        img = cv2.resize(img, (541, 416), interpolation=cv2.INTER_AREA)
                        res = cv2.imencode('.JPEG', img)[1].tobytes()
                        vidcap.release()
                        return web.Response(body=res)
                    else:
                        print(fileName,"server getFileImag  can not open image=========")
                        return web.json_response({"error":["server","getFileImg"]})
        except:
            print("can not open file")
            return web.json_response({"error":["server","getFileImg","can not open file"]})

    async def on_shutdown(self, app):
        print("start on_shutdown")
        if  'websocketscmd' in app:
            wss = list(self.app['websocketscmd'])
            for ws in wss:
                try:
                    await ws.close(code=1001, message='Server shutdown')
                finally:
                    self.app['websocketscmd'].discard(ws)

        #  background task 

    async def background_process(self):
        #await self.try_make_db()
        while True:
            # self.log.debug('Run background task each 1 min')
            # print("len websocketscmd:", str(len(self.app['websocketscmd'])))
            try:
                #await save_statistic("intersetions", "file_0", ["border_a","border_b"])
                # print("server tik")
                if 'websocketscmd' in self.app:
                    if self.app['websocketscmd']:
                        if 'manager' in self.app:
                            res = self.app['manager'].getSreamsStatus()
                            #print("res", len(res), res)
                            if any(res):
                                #print("res 2")
                                res = {"camsList": res}
                                try:
                                    for ws in self.app['websocketscmd']:
                                        await ws.send_json(res)
                                except:
                                    print("Unexpected error:", sys.exc_info()[0])
            except:
                print('server loop Errror')
                print(sys.exc_info())
            await asyncio.sleep(1)

    async def start_background_tasks(self, app):
        app['dispatch'] = asyncio.create_task(self.background_process())

    async def cleanup_background_tasks(self, app):
        print("start cleanup_background_tasks")
        app['dispatch'].cancel()
        await self.app['dispatch']
      #  end background task 

if __name__ == "__main__":
    try:
        server = Server()
    except:
        print("stop bu exception")
    print("server stooped Ok")