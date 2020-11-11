#! /usr/bin/env python
import asyncio
# import aiohttp_debugtoolbar
from aiohttp_session import session_middleware
# from aiohttp_session.cookie_storage import EncryptedCookieStorage

from aiohttp import web
from typing import Any, AsyncIterator, Awaitable, Callable, Dict
import aiosqlite
from pathlib import Path
from routes import routes
from middlewares import authorize
# from motor import motor_asyncio as ma
# import asyncio
import io
import sqlite3

import settings
import server_data
import hashlib
import camera
import ssl

import logging

log = logging.getLogger('app')
log.setLevel(logging.DEBUG)
f = logging.Formatter('[L:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(f)
log.addHandler(ch)

async def on_shutdown(app):
    for ws in app['websockets']:
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
    app.router.add_route(route[0], route[1], route[2], name=route[3])
app['static_root_url'] = '/static'
app.router.add_static('/static', 'static', name='static')
# app.router.add_static('/', 'index', name='static')
# end route part

#  background task 
async def background_process():
    while True:
        log.debug('Run background task each 1 min')
        await asyncio.sleep(60)
async def start_background_tasks(app):
    app['dispatch'] = asyncio.create_task(background_process())
async def cleanup_background_tasks(app):
    app['dispatch'].cancel()
    await app['dispatch']
app.on_startup.append(start_background_tasks)
app.on_cleanup.append(cleanup_background_tasks)
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
app['websockets'] = []

#  start Camera Object
cam = camera.Camera("testCamera", log)
cam.start()
# time.sleep(10)

# log.info('The server running...')
web.run_app(app)
#  Stop cameta Object
cam.kill()
cam.join()
log.info('The server stopped!')
