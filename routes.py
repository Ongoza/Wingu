from views.websocket import WebSocket
from views.auth import Sign


routes = [
    ('GET', '/ws',      WebSocket, 'websocket'),
    # ('*',   '/login',   Login,     'login'),
    ('POST', '/sign/{action}',  Sign,    'sign'),
    # ('GET',  '/signout', SignOut,   'signout'),
]
