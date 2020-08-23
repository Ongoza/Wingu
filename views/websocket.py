from datetime import datetime

from aiohttp_session import get_session
from aiohttp import web, WSMsgType
# from server import log

class WebSocket(web.View):
    async def get(self):
        ws = web.WebSocketResponse()
        await ws.prepare(self.request)        
        session = await get_session(self.request)
        print("session chat", session)
        login = session.get('user')
        if login is not None:
            for _ws in self.request.app['websockets']:
                _ws.send_str('%s joined' % login)
            self.request.app['websockets'].append(ws)
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    if msg.data == 'close':
                        await ws.close()
                    else:
                        # message = Message(self.request.app.db)
                        # result = await message.save(user=login, msg=msg.data)
                        print(msg.data)
                        for _ws in self.request.app['websockets']:
                            await _ws.send_str('{"user": "%s", "msg": "%s"}' % (login, msg.data))
                elif msg.type == WSMsgType.ERROR:
                    print('ws connection closed with exception %s' % ws.exception())

            self.request.app['websockets'].remove(ws)
            for _ws in self.request.app['websockets']:
                _ws.send_str('%s disconected' % login)
            print('websocket connection closed')
            return ws
        else: 
            print("Wrong user login")
            return False